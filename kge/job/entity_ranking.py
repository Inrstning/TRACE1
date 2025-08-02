import math
import time

import torch
import kge.job
from kge.job import EvaluationJob, Job
from kge import Config, Dataset
from collections import defaultdict

from kge.util.time_aware_filter import filter_entities_by_time, filter_scores_by_time


class EntityRankingJob(EvaluationJob):
    """ Entity ranking evaluation protocol """

    def __init__(self, config: Config, dataset: Dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)
        self.config.check(
            "eval.tie_handling",
            ["rounded_mean_rank", "best_rank", "worst_rank"],
        )
        self.tie_handling = self.config.get("eval.tie_handling")
        self.is_prepared = False
        
        # Time-aware filtering parameters
        self.time_aware_filtering = self.config.get("eval.time_aware_filtering")
        self.filter_with_same_timestamp = self.config.get("eval.filter_with_same_timestamp")
        self.time_aware_filter_data = None  # Store time-aware filtering index data

        if self.__class__ == EntityRankingJob:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        """Construct all indexes needed to run."""

        if self.is_prepared:
            return

        # create data and precompute indexes
        self.triples = self.dataset.split(self.config.get("eval.split"))
        for split in self.filter_splits:
            self.dataset.index(f"{split}_sp_to_o")
            self.dataset.index(f"{split}_po_to_s")
        if "test" not in self.filter_splits and self.filter_with_test:
            self.dataset.index("test_sp_to_o")
            self.dataset.index("test_po_to_s")
            

        if self.time_aware_filtering:
            self.config.log("Preparing time-aware filtering data...")
            
            self.time_aware_filter_data = {} 
            self.time_aware_filter_data_with_test = {}  
            
            standard_filter_triples = []
            for split in self.filter_splits:
                split_triples = self.dataset.split(split)
                standard_filter_triples.append(split_triples)
                
            if "test" not in self.filter_splits and self.filter_with_test:
                test_triples = self.dataset.split("test")
                
                all_filter_triples = standard_filter_triples.copy()
                all_filter_triples.append(test_triples)
            
                if all_filter_triples:
                    all_combined_triples = torch.cat(all_filter_triples)

                    self.time_aware_filter_data_with_test = filter_entities_by_time(
                        all_combined_triples, 
                        None, 
                        self.dataset.num_entities(),
                        "both",
                        self.filter_with_same_timestamp
                    )
                    self.config.log(f"Created time-aware filter with test data: {len(self.time_aware_filter_data_with_test)} timestamp-specific entries")
            

            if standard_filter_triples:
                standard_combined_triples = torch.cat(standard_filter_triples)

                self.time_aware_filter_data = filter_entities_by_time(
                    standard_combined_triples, 
                    None, 
                    self.dataset.num_entities(),
                    "both",
                    self.filter_with_same_timestamp
                )
                self.config.log(f"Created standard time-aware filter: {len(self.time_aware_filter_data)} timestamp-specific entries (filter_with_same_timestamp={self.filter_with_same_timestamp})")


        self.loader = torch.utils.data.DataLoader(
            self.triples,
            collate_fn=self._collate,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.config.get("eval.num_workers"),
            pin_memory=self.config.get("eval.pin_memory"),
        )

        self.model.prepare_job(self)
        self.is_prepared = True

    def _collate(self, batch):
        "Looks up true triples for each triple in the batch"
        label_coords = []
        for split in self.filter_splits:
            split_label_coords = kge.job.util.get_sp_po_coords_from_spo_batch(
                batch,
                self.dataset.num_entities(),
                self.dataset.index(f"{split}_sp_to_o"),
                self.dataset.index(f"{split}_po_to_s"),
            )
            label_coords.append(split_label_coords)

        label_coords_standard = torch.cat(label_coords)


        label_coords_with_test = label_coords_standard.clone()
        if "test" not in self.filter_splits and self.filter_with_test:
            test_label_coords = kge.job.util.get_sp_po_coords_from_spo_batch(
                batch,
                self.dataset.num_entities(),
                self.dataset.index("test_sp_to_o"),
                self.dataset.index("test_po_to_s"),
            )

            label_coords_with_test = torch.cat([label_coords_with_test, test_label_coords])


        batch = torch.cat(batch).reshape((-1, 4))

        return batch, label_coords_standard, label_coords_with_test

    @torch.no_grad()
    def run(self) -> dict:
        self._prepare()

        was_training = self.model.training
        self.model.eval()
        self.config.log(
            "Evaluating on "
            + self.eval_split
            + " data (epoch {})...".format(self.epoch)
        )
        num_entities = self.dataset.num_entities()


        if hasattr(self.model, "_scorer") and self.model.__class__.__name__ == "Trace":

            self.model._scorer._entity_embedder = self.model.get_s_embedder
            self.model._scorer._relation_embedder = self.model.get_p_embedder
            self.model._scorer._time_embedder = self.model.get_t_embedder

        filter_with_test = "test" not in self.filter_splits and self.filter_with_test


        rankings = (
            ["_raw", "_filt", "_filt_test"] if filter_with_test else ["_raw", "_filt"]
        )
        

        if self.time_aware_filtering:
            rankings.append("_time_aware_filt")
            if filter_with_test:
                rankings.append("_time_aware_filt_test") 


        labels_for_ranking = defaultdict(lambda: None)

        hists = dict()
        hists_filt = dict()
        hists_filt_test = dict()
        
        if self.time_aware_filtering:
            hists_time_aware_filt = dict()
            if filter_with_test:
                hists_time_aware_filt_test = dict()


        epoch_time = -time.time()
        for batch_number, batch_coords in enumerate(self.loader):

            batch = batch_coords[0].to(self.device)
            s, p, o, t = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3]
            label_coords = batch_coords[1].to(self.device) # Standard filtering coordinates
            if filter_with_test:
                label_coords_with_test = batch_coords[2].to(self.device) 
                test_labels = kge.job.util.coord_to_sparse_tensor(
                    len(batch),
                    2 * num_entities,
                    label_coords_with_test, 
                    self.device,
                    float("Inf"),
                )
                labels_for_ranking["_filt_test"] = test_labels

            labels = kge.job.util.coord_to_sparse_tensor(
                len(batch), 2 * num_entities, label_coords, self.device, float("Inf")
            )
            labels_for_ranking["_filt"] = labels

            o_true_scores = self.model.score_spo(s, p, o, t, "o").view(-1)
            s_true_scores = self.model.score_spo(s, p, o, t, "s").view(-1)


            ranks_and_ties_for_ranking = defaultdict(
                lambda: [
                    torch.zeros(s.size(0), dtype=torch.long).to(self.device),
                    torch.zeros(s.size(0), dtype=torch.long).to(self.device),
                ]
            )


            if self.config.get("eval.chunk_size") > -1:
                chunk_size = self.config.get("eval.chunk_size")
            else:
                chunk_size = self.dataset.num_entities()

            for chunk_number in range(math.ceil(num_entities / chunk_size)):
                chunk_start = chunk_size * chunk_number
                chunk_end = min(chunk_size * (chunk_number + 1), num_entities)

                if chunk_size == self.dataset.num_entities():
                    scores = self.model.score_sp_po(s, p, o, t, None)
                else:
                    scores = self.model.score_sp_po(
                        s, p, o, t, torch.arange(chunk_start, chunk_end).to(self.device)
                    )
                scores_sp = scores[:, : chunk_end - chunk_start]
                scores_po = scores[:, chunk_end - chunk_start :]

                s_in_chunk_mask = (chunk_start <= s) & (s < chunk_end)
                o_in_chunk_mask = (chunk_start <= o) & (o < chunk_end)
                o_in_chunk = (o[o_in_chunk_mask] - chunk_start).long()
                s_in_chunk = (s[s_in_chunk_mask] - chunk_start).long()
                scores_sp[o_in_chunk_mask, o_in_chunk] = o_true_scores[o_in_chunk_mask]
                scores_po[s_in_chunk_mask, s_in_chunk] = s_true_scores[s_in_chunk_mask]

                # now compute the rankings (assumes order: None, _filt, _filt_test)
                for ranking in rankings:
                    if labels_for_ranking[ranking] is None:
                        labels_chunk = None
                    else:
                        # densify the needed part of the sparse labels tensor
                        labels_chunk = self._densify_chunk_of_labels(
                            labels_for_ranking[ranking], chunk_start, chunk_end
                        )

                        # remove current example from labels
                        labels_chunk[o_in_chunk_mask, o_in_chunk] = 0
                        labels_chunk[
                            s_in_chunk_mask, s_in_chunk + (chunk_end - chunk_start)
                        ] = 0


                    if ranking in ["_time_aware_filt", "_time_aware_filt_test"] and self.time_aware_filtering:
                        scores_sp_time_filt = scores_sp.clone()
                        scores_po_time_filt = scores_po.clone()
                        batch_sp = torch.cat([
                            s.view(-1, 1),
                            p.view(-1, 1),
                            o.view(-1, 1),
                            t.view(-1, 1)
                        ], dim=1)
                        
                        filtered_count_sp = 0
                        filtered_count_po = 0
                        
                        filter_data = self.time_aware_filter_data_with_test if ranking == "_time_aware_filt_test" else self.time_aware_filter_data
                        
                        for i in range(len(batch_sp)):
                            key_sp = (int(batch_sp[i, 0]), int(batch_sp[i, 1]), int(batch_sp[i, 3]))
                            if key_sp in filter_data:
                                filtered_entities_in_chunk = 0
                                for o_idx in filter_data[key_sp]:
                                    if chunk_start <= o_idx < chunk_end and o_idx != batch_sp[i, 2]:
                                        scores_sp_time_filt[i, o_idx - chunk_start] = float('-inf')
                                        filtered_entities_in_chunk += 1
                                filtered_count_sp += filtered_entities_in_chunk
                            
                            key_po = (int(batch_sp[i, 2]), int(batch_sp[i, 1]), int(batch_sp[i, 3]))
                            if key_po in filter_data:
                                filtered_entities_in_chunk = 0
                                for s_idx in filter_data[key_po]:
                                    if chunk_start <= s_idx < chunk_end and s_idx != batch_sp[i, 0]:
                                        scores_po_time_filt[i, s_idx - chunk_start] = float('-inf')
                                        filtered_entities_in_chunk += 1
                                filtered_count_po += filtered_entities_in_chunk
                        
                        if batch_number == 0 and chunk_number == 0:
                            filter_type = "with test" if ranking == "_time_aware_filt_test" else "standard"
                            self.config.log(f"Time-aware filtering ({filter_type}): filtered {filtered_count_sp} tail entities and {filtered_count_po} head entities in first batch/chunk")
                        

                        s_rank_chunk, s_num_ties_chunk = self._get_ranks_and_num_ties(scores_po_time_filt, s_true_scores)
                        o_rank_chunk, o_num_ties_chunk = self._get_ranks_and_num_ties(scores_sp_time_filt, o_true_scores)
                    else:

                        (
                            s_rank_chunk,
                            s_num_ties_chunk,
                            o_rank_chunk,
                            o_num_ties_chunk,

                            scores_sp_filt,
                            scores_po_filt 
                        ) = self._filter_and_rank(
                            scores_sp, scores_po, labels_chunk, o_true_scores, s_true_scores
                        )
                        

                        if ranking != "_time_aware_filt": 
                           scores_sp = scores_sp_filt
                           scores_po = scores_po_filt


                    ranks_and_ties_for_ranking["s" + ranking][0] += s_rank_chunk
                    ranks_and_ties_for_ranking["s" + ranking][1] += s_num_ties_chunk
                    ranks_and_ties_for_ranking["o" + ranking][0] += o_rank_chunk
                    ranks_and_ties_for_ranking["o" + ranking][1] += o_num_ties_chunk

            s_ranks = self._get_ranks(
                ranks_and_ties_for_ranking["s_raw"][0],
                ranks_and_ties_for_ranking["s_raw"][1],
            )
            o_ranks = self._get_ranks(
                ranks_and_ties_for_ranking["o_raw"][0],
                ranks_and_ties_for_ranking["o_raw"][1],
            )
            s_ranks_filt = self._get_ranks(
                ranks_and_ties_for_ranking["s_filt"][0],
                ranks_and_ties_for_ranking["s_filt"][1],
            )
            o_ranks_filt = self._get_ranks(
                ranks_and_ties_for_ranking["o_filt"][0],
                ranks_and_ties_for_ranking["o_filt"][1],
            )
            

            if self.time_aware_filtering:
                s_ranks_time_aware_filt = self._get_ranks(
                    ranks_and_ties_for_ranking["s_time_aware_filt"][0],
                    ranks_and_ties_for_ranking["s_time_aware_filt"][1],
                )
                o_ranks_time_aware_filt = self._get_ranks(
                    ranks_and_ties_for_ranking["o_time_aware_filt"][0],
                    ranks_and_ties_for_ranking["o_time_aware_filt"][1],
                )
                

                if filter_with_test:
                    s_ranks_time_aware_filt_test = self._get_ranks(
                        ranks_and_ties_for_ranking["s_time_aware_filt_test"][0],
                        ranks_and_ties_for_ranking["s_time_aware_filt_test"][1],
                    )
                    o_ranks_time_aware_filt_test = self._get_ranks(
                        ranks_and_ties_for_ranking["o_time_aware_filt_test"][0],
                        ranks_and_ties_for_ranking["o_time_aware_filt_test"][1],
                    )


            batch_hists = dict()
            batch_hists_filt = dict()
            for f in self.hist_hooks:
                f(batch_hists, s, p, o, s_ranks, o_ranks, job=self)
                f(batch_hists_filt, s, p, o, s_ranks_filt, o_ranks_filt, job=self)
                

            if self.time_aware_filtering:
                batch_hists_time_aware_filt = dict()
                for f in self.hist_hooks:
                    f(batch_hists_time_aware_filt, s, p, o, s_ranks_time_aware_filt, o_ranks_time_aware_filt, job=self)
                

                if filter_with_test:
                    batch_hists_time_aware_filt_test = dict()
                    for f in self.hist_hooks:
                        f(batch_hists_time_aware_filt_test, s, p, o, s_ranks_time_aware_filt_test, o_ranks_time_aware_filt_test, job=self)


            if filter_with_test:
                batch_hists_filt_test = dict()
                s_ranks_filt_test = self._get_ranks(
                    ranks_and_ties_for_ranking["s_filt_test"][0],
                    ranks_and_ties_for_ranking["s_filt_test"][1],
                )
                o_ranks_filt_test = self._get_ranks(
                    ranks_and_ties_for_ranking["o_filt_test"][0],
                    ranks_and_ties_for_ranking["o_filt_test"][1],
                )
                for f in self.hist_hooks:
                    f(
                        batch_hists_filt_test,
                        s,
                        p,
                        o,
                        s_ranks_filt_test,
                        o_ranks_filt_test,
                        job=self,
                    )


            if self.trace_examples:
                entry = {
                    "type": "entity_ranking",
                    "scope": "example",
                    "split": self.eval_split,
                    "filter_splits": self.filter_splits,
                    "size": len(batch),
                    "batches": len(self.loader),
                    "epoch": self.epoch,
                }
                for i in range(len(batch)):
                    entry["batch"] = i
                    entry["s"], entry["p"], entry["o"], entry["t"] = (
                        s[i].item(),
                        p[i].item(),
                        o[i].item(),
                        t[i].item(),
                    )
                    

                    if self.time_aware_filtering:
                        entry["rank_time_aware_filtered"] = (
                            o_ranks_time_aware_filt[i].item() + 1
                        )
                        if filter_with_test:
                            entry["rank_time_aware_filtered_with_test"] = (
                                o_ranks_time_aware_filt_test[i].item() + 1
                            )
                    self.trace(
                        event="example_rank",
                        task="sp",
                        rank=o_ranks[i].item() + 1,
                        rank_filtered=o_ranks_filt[i].item() + 1,
                        **entry,
                    )
                    

                    if self.time_aware_filtering:
                        entry["rank_time_aware_filtered"] = (
                            s_ranks_time_aware_filt[i].item() + 1
                        )
                        if filter_with_test:
                            entry["rank_time_aware_filtered_with_test"] = (
                                s_ranks_time_aware_filt_test[i].item() + 1
                            )
                    self.trace(
                        event="example_rank",
                        task="po",
                        rank=s_ranks[i].item() + 1,
                        rank_filtered=s_ranks_filt[i].item() + 1,
                        **entry,
                    )


            metrics = self._compute_metrics(batch_hists["all"])
            metrics.update(
                self._compute_metrics(batch_hists_filt["all"], suffix="_filtered")
            )
            if filter_with_test:
                metrics.update(
                    self._compute_metrics(
                        batch_hists_filt_test["all"], suffix="_filtered_with_test"
                    )
                )
                

            if self.time_aware_filtering:
                metrics.update(
                    self._compute_metrics(
                        batch_hists_time_aware_filt["all"], suffix="_time_aware_filtered"
                    )
                )
                if filter_with_test:
                    metrics.update(
                        self._compute_metrics(
                            batch_hists_time_aware_filt_test["all"], suffix="_time_aware_filtered_with_test"
                        )
                    )


            if self.trace_batch:
                self.trace(
                    event="batch_completed",
                    type="entity_ranking",
                    scope="batch",
                    split=self.eval_split,
                    filter_splits=self.filter_splits,
                    epoch=self.epoch,
                    batch=batch_number,
                    size=len(batch),
                    batches=len(self.loader),
                    **metrics,
                )

            if self.time_aware_filtering:
                display_format = (
                    "\r"  # go back
                    + "{}  batch:{: "
                    + str(1 + int(math.ceil(math.log10(len(self.loader)))))
                    + "d}/{}, mrr (filt./time-filt.): {:4.3f} ({:4.3f}/{:4.3f}), "
                    + "hits@1: {:4.3f} ({:4.3f}/{:4.3f}), "
                    + "hits@{}: {:4.3f} ({:4.3f}/{:4.3f})"
                    + "\033[K"  # clear to right
                )
                self.config.print(
                    display_format.format(
                        self.config.log_prefix,
                        batch_number,
                        len(self.loader) - 1,
                        metrics["mean_reciprocal_rank"],
                        metrics["mean_reciprocal_rank_filtered"],
                        metrics["mean_reciprocal_rank_time_aware_filtered"],
                        metrics["hits_at_1"],
                        metrics["hits_at_1_filtered"],
                        metrics["hits_at_1_time_aware_filtered"],
                        self.hits_at_k_s[-1],
                        metrics["hits_at_{}".format(self.hits_at_k_s[-1])],
                        metrics["hits_at_{}_filtered".format(self.hits_at_k_s[-1])],
                        metrics["hits_at_{}_time_aware_filtered".format(self.hits_at_k_s[-1])],
                    ),
                    end="",
                    flush=True,
                )
            else:

                self.config.print(
                    (
                        "\r"  
                        + "{}  batch:{: "
                        + str(1 + int(math.ceil(math.log10(len(self.loader)))))
                        + "d}/{}, mrr (filt.): {:4.3f} ({:4.3f}), "
                        + "hits@1: {:4.3f} ({:4.3f}), "
                        + "hits@{}: {:4.3f} ({:4.3f})"
                        + "\033[K"  # clear to right
                    ).format(
                        self.config.log_prefix,
                        batch_number,
                        len(self.loader) - 1,
                        metrics["mean_reciprocal_rank"],
                        metrics["mean_reciprocal_rank_filtered"],
                        metrics["hits_at_1"],
                        metrics["hits_at_1_filtered"],
                        self.hits_at_k_s[-1],
                        metrics["hits_at_{}".format(self.hits_at_k_s[-1])],
                        metrics["hits_at_{}_filtered".format(self.hits_at_k_s[-1])],
                    ),
                    end="",
                    flush=True,
                )

            def merge_hist(target_hists, source_hists):
                for key, hist in source_hists.items():
                    if key in target_hists:
                        target_hists[key] = target_hists[key] + hist
                    else:
                        target_hists[key] = hist

            merge_hist(hists, batch_hists)
            merge_hist(hists_filt, batch_hists_filt)
            if filter_with_test:
                merge_hist(hists_filt_test, batch_hists_filt_test)
                

            if self.time_aware_filtering:
                merge_hist(hists_time_aware_filt, batch_hists_time_aware_filt)
                if filter_with_test:
                    merge_hist(hists_time_aware_filt_test, batch_hists_time_aware_filt_test)


        self.config.print("\033[2K\r", end="", flush=True)  # clear line and go back
        for key, hist in hists.items():
            name = "_" + key if key != "all" else ""
            metrics.update(self._compute_metrics(hists[key], suffix=name))
            metrics.update(
                self._compute_metrics(hists_filt[key], suffix="_filtered" + name)
            )
            if filter_with_test:
                metrics.update(
                    self._compute_metrics(
                        hists_filt_test[key], suffix="_filtered_with_test" + name
                    )
                )
                

            if self.time_aware_filtering:
                metrics.update(
                    self._compute_metrics(
                        hists_time_aware_filt[key], suffix="_time_aware_filtered" + name
                    )
                )
                if filter_with_test:
                    metrics.update(
                        self._compute_metrics(
                            hists_time_aware_filt_test[key], suffix="_time_aware_filtered_with_test" + name
                        )
                    )
                
        epoch_time += time.time()

        # compute trace
        trace_entry = dict(
            type="entity_ranking",
            scope="epoch",
            split=self.eval_split,
            filter_splits=self.filter_splits,
            epoch=self.epoch,
            size=len(self.triples),
            batches=len(self.loader),
            epoch_time=epoch_time,
            event="eval_completed",
            **metrics,
        )


        if self.time_aware_filtering:
            self.config.log(
                f"Time-aware filtering stats: {len(self.time_aware_filter_data)} timestamp-specific entries"
            )
            

            self.config.log("================= EVALUATION METRICS =================")
            self.config.log(f"Mean Reciprocal Rank:")
            self.config.log(f"  Raw: {metrics['mean_reciprocal_rank']:.6f}")
            self.config.log(f"  Filtered: {metrics['mean_reciprocal_rank_filtered']:.6f}")
            if filter_with_test:
                self.config.log(f"  Filtered with test: {metrics['mean_reciprocal_rank_filtered_with_test']:.6f}")
            self.config.log(f"  Time-aware filtered: {metrics['mean_reciprocal_rank_time_aware_filtered']:.6f}")
            if filter_with_test:
                self.config.log(f"  Time-aware filtered with test: {metrics['mean_reciprocal_rank_time_aware_filtered_with_test']:.6f}")
            

            mrr_diff = metrics['mean_reciprocal_rank_time_aware_filtered'] - metrics['mean_reciprocal_rank_filtered']
            mrr_diff_percent = (mrr_diff / metrics['mean_reciprocal_rank_filtered']) * 100
            self.config.log(f"  Time-aware vs Standard: {mrr_diff:.6f} ({mrr_diff_percent:+.2f}%)")
            
            if filter_with_test:

                mrr_diff_test = metrics['mean_reciprocal_rank_time_aware_filtered_with_test'] - metrics['mean_reciprocal_rank_filtered_with_test']
                mrr_diff_test_percent = (mrr_diff_test / metrics['mean_reciprocal_rank_filtered_with_test']) * 100
                self.config.log(f"  Time-aware vs Filtered with test: {mrr_diff_test:.6f} ({mrr_diff_test_percent:+.2f}%)")
                

                mrr_diff_time = metrics['mean_reciprocal_rank_time_aware_filtered_with_test'] - metrics['mean_reciprocal_rank_time_aware_filtered']
                mrr_diff_time_percent = (mrr_diff_time / metrics['mean_reciprocal_rank_time_aware_filtered']) * 100
                self.config.log(f"  Time-aware with test vs Time-aware standard: {mrr_diff_time:.6f} ({mrr_diff_time_percent:+.2f}%)")
            

            for k in self.hits_at_k_s:
                self.config.log(f"Hits@{k}:")
                self.config.log(f"  Raw: {metrics[f'hits_at_{k}']:.6f}")
                self.config.log(f"  Filtered: {metrics[f'hits_at_{k}_filtered']:.6f}")
                if filter_with_test:
                    self.config.log(f"  Filtered with test: {metrics[f'hits_at_{k}_filtered_with_test']:.6f}")
                self.config.log(f"  Time-aware filtered: {metrics[f'hits_at_{k}_time_aware_filtered']:.6f}")
                if filter_with_test:
                    self.config.log(f"  Time-aware filtered with test: {metrics[f'hits_at_{k}_time_aware_filtered_with_test']:.6f}")
                

                hits_diff = metrics[f'hits_at_{k}_time_aware_filtered'] - metrics[f'hits_at_{k}_filtered']
                hits_diff_percent = (hits_diff / metrics[f'hits_at_{k}_filtered']) * 100 if metrics[f'hits_at_{k}_filtered'] > 0 else 0
                self.config.log(f"  Time-aware vs Standard: {hits_diff:.6f} ({hits_diff_percent:+.2f}%)")
                
                if filter_with_test:

                    hits_diff_test = metrics[f'hits_at_{k}_time_aware_filtered_with_test'] - metrics[f'hits_at_{k}_filtered_with_test']
                    hits_diff_test_percent = (hits_diff_test / metrics[f'hits_at_{k}_filtered_with_test']) * 100 if metrics[f'hits_at_{k}_filtered_with_test'] > 0 else 0
                    self.config.log(f"  Time-aware with test vs Filtered with test: {hits_diff_test:.6f} ({hits_diff_test_percent:+.2f}%)")
                    

                    hits_diff_time = metrics[f'hits_at_{k}_time_aware_filtered_with_test'] - metrics[f'hits_at_{k}_time_aware_filtered']
                    hits_diff_time_percent = (hits_diff_time / metrics[f'hits_at_{k}_time_aware_filtered']) * 100 if metrics[f'hits_at_{k}_time_aware_filtered'] > 0 else 0
                    self.config.log(f"  Time-aware with test vs Time-aware standard: {hits_diff_time:.6f} ({hits_diff_time_percent:+.2f}%)")
            
            self.config.log("====================================================")
            


        self.config.print("")
        main_metrics = [
            (name, value)
            for name, value in metrics.items()
            if (
                (
                    name.startswith("mean_reciprocal_rank") or name.startswith("hits_at_")
                )
                and not (
                    name.startswith("mean_reciprocal_rank_raw_")
                    or name.startswith("hits_at_")
                    and "_raw_" in name
                )
            )
        ]
        for name, value in main_metrics:
            self.config.print(f"{name}: {value:12.6f}")

        if not was_training:
            self.model.eval()
        else:
            self.model.train()

        return trace_entry

    def _densify_chunk_of_labels(
        self, labels: torch.Tensor, chunk_start: int, chunk_end: int
    ) -> torch.Tensor:
        """Creates a dense chunk of a sparse label tensor.

        A chunk here is a range of entity values with 'chunk_start' being the lower
        bound and 'chunk_end' the upper bound.

        The resulting tensor contains the labels for the sp chunk and the po chunk.

        :param labels: sparse tensor containing the labels corresponding to the batch
        for sp and po

        :param chunk_start: int start index of the chunk

        :param chunk_end: int end index of the chunk

        :return: batch_size x chunk_size*2 dense tensor with labels for the sp chunk and
        the po chunk.

        """
        num_entities = self.dataset.num_entities()
        indices = labels._indices()
        mask_sp = (chunk_start <= indices[1, :]) & (indices[1, :] < chunk_end)
        mask_po = ((chunk_start + num_entities) <= indices[1, :]) & (
            indices[1, :] < (chunk_end + num_entities)
        )
        indices_sp_chunk = indices[:, mask_sp]
        indices_sp_chunk[1, :] = indices_sp_chunk[1, :] - chunk_start
        indices_po_chunk = indices[:, mask_po]
        indices_po_chunk[1, :] = (
            indices_po_chunk[1, :] - num_entities - chunk_start * 2 + chunk_end
        )
        indices_chunk = torch.cat((indices_sp_chunk, indices_po_chunk), dim=1)
        dense_labels = torch.sparse.LongTensor(
            indices_chunk,
            labels._values()[mask_sp | mask_po],
            torch.Size([labels.size()[0], (chunk_end - chunk_start) * 2]),
        ).to_dense()
        return dense_labels

    def _filter_and_rank(
        self,
        scores_sp: torch.Tensor,
        scores_po: torch.Tensor,
        labels: torch.Tensor,
        o_true_scores: torch.Tensor,
        s_true_scores: torch.Tensor,
    ):
        """Filters the current examples with the given labels and returns counts rank and
num_ties for each true score.

        :param scores_sp: batch_size x chunk_size tensor of scores

        :param scores_po: batch_size x chunk_size tensor of scores

        :param labels: batch_size x 2*chunk_size tensor of scores

        :param o_true_scores: batch_size x 1 tensor containing the scores of the actual
        objects in batch

        :param s_true_scores: batch_size x 1 tensor containing the scores of the actual
        subjects in batch

        :return: batch_size x 1 tensors rank and num_ties for s and o and filtered
        scores_sp and scores_po

        """
        chunk_size = scores_sp.shape[1]
        if labels is not None:
            # remove current example from labels
            labels_sp = labels[:, :chunk_size]
            labels_po = labels[:, chunk_size:]
            scores_sp = scores_sp - labels_sp
            scores_po = scores_po - labels_po
        o_rank, o_num_ties = self._get_ranks_and_num_ties(scores_sp, o_true_scores)
        s_rank, s_num_ties = self._get_ranks_and_num_ties(scores_po, s_true_scores)
        return s_rank, s_num_ties, o_rank, o_num_ties, scores_sp, scores_po

    @staticmethod
    def _get_ranks_and_num_ties(
        scores: torch.Tensor, true_scores: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        """Returns rank and number of ties of each true score in scores.

        :param scores: batch_size x entities tensor of scores

        :param true_scores: batch_size x 1 tensor containing the actual scores of the batch

        :return: batch_size x 1 tensors rank and num_ties
        """
        # process NaN values
        scores = scores.clone()
        scores[torch.isnan(scores)] = float("-Inf")
        true_scores = true_scores.clone()
        true_scores[torch.isnan(true_scores)] = float("-Inf")

        # Determine how many scores are greater than / equal to each true answer (in its
        # corresponding row of scores)
        rank = torch.sum(scores > true_scores.view(-1, 1), dim=1, dtype=torch.long)
        num_ties = torch.sum(scores == true_scores.view(-1, 1), dim=1, dtype=torch.long)
        return rank, num_ties

    def _get_ranks(self, rank: torch.Tensor, num_ties: torch.Tensor) -> torch.Tensor:
        """Calculates the final rank from (minimum) rank and number of ties.

        :param rank: batch_size x 1 tensor with number of scores greater than the one of
        the true score

        :param num_ties: batch_size x tensor with number of scores equal as the one of
        the true score

        :return: batch_size x 1 tensor of ranks

        """

        if self.tie_handling == "rounded_mean_rank":
            return rank + num_ties // 2
        elif self.tie_handling == "best_rank":
            return rank
        elif self.tie_handling == "worst_rank":
            return rank + num_ties - 1
        else:
            raise NotImplementedError

    def _compute_metrics(self, rank_hist, suffix=""):
        """Computes desired matrix from rank histogram"""
        metrics = {}
        n = torch.sum(rank_hist).item()

        ranks = torch.arange(1, self.dataset.num_entities() + 1).float().to(self.device)
        metrics["mean_rank" + suffix] = (
            (torch.sum(rank_hist * ranks).item() / n) if n > 0.0 else 0.0
        )

        reciprocal_ranks = 1.0 / ranks
        metrics["mean_reciprocal_rank" + suffix] = (
            (torch.sum(rank_hist * reciprocal_ranks).item() / n) if n > 0.0 else 0.0
        )

        hits_at_k = (
            (torch.cumsum(rank_hist[: max(self.hits_at_k_s)], dim=0) / n).tolist()
            if n > 0.0
            else [0.0] * max(self.hits_at_k_s)
        )

        for i, k in enumerate(self.hits_at_k_s):
            metrics["hits_at_{}{}".format(k, suffix)] = hits_at_k[k - 1]

        return metrics
