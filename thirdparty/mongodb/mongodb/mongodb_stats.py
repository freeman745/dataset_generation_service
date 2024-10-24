from typing import Optional, List, Dict, Set
import os
import datetime
from collections import defaultdict
import statistics
import json
from . import MongoDBItem, MongoDBReader
from .constants import (
    StatMode,
    StatDurations,
    OID_MAPS_DIR,
    OidMapsKeys,
    GeneralKeys as GKeys,
    ClassNames as CNs,
    OrderManagerNodeKeys as OmKeys,
    ErrorCategoriesNames as ECNames,
    ActionKeys as AcKeys,
)
from .utils import (
    draw_stats,
    exception_handler,
    get_agregated_duration,
    get_cls_attributes,
    memoize,
)

from .utils.decorators import METHODS_LOGS_KEY


class MongoDBStats:
    def __init__(
        self, items: Optional[List[MongoDBItem]] = None, hash_id: Optional[str] = None
    ):
        if items is not None:
            self.items = items
        elif hash_id is not None:
            self.items = self._get_items_from_hash(hash_id)

    def set_items(self, items: List[MongoDBItem]):
        self.items = items

    @memoize
    @property
    def successful_items(self) -> List[MongoDBItem]:
        return [item for item in self.items if item.is_successful]

    @memoize
    @property
    def successful_items_num(self) -> int:
        return len(self.successful_items)

    @property
    def failed_items(self) -> List[MongoDBItem]:
        return [item for item in self.items if not item.is_successful]

    @memoize
    @property
    def failed_items_num(self) -> int:
        return len(self.failed_items)

    @property
    def multi_picks_items(self) -> List[MongoDBItem]:
        return [item for item in self.items if item.multi_picks]

    @memoize
    @property
    def multi_picks_items_num(self) -> int:
        return len(self.multi_picks_items)

    @memoize
    @property
    def git_shas(self) -> Set[str]:
        return set([item.git_sha for item in self.items])

    @memoize
    @property
    def git_branches(self) -> Set[str]:
        return set(
            [item.git_branch for item in self.items if item.git_branch is not None]
        )

    @memoize
    @property
    def start_time(self) -> Optional[datetime.datetime]:
        start_times = [item.start_time for item in self.items if item.start_time]
        if len(start_times) == 0:
            return None
        return min(start_times)

    @memoize
    @property
    def end_time(self) -> Optional[datetime.datetime]:
        end_times = [item.end_time for item in self.items if item.end_time]
        if len(end_times) == 0:
            return None
        return max(end_times)

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        # return self.get_summary()
        desc = f"{self.__class__.__name__} {len(self)} items"
        desc += f", success rate: {self.get_success_rate()}"
        return desc

    @memoize
    def get_actions_durations(self, mode=StatMode.mean) -> Dict[str, List[float]]:
        data = defaultdict(list)
        for item in self.items:
            for action in item.get_actions():
                data[action.full_name].append(action.duration)
        for action, values in data.items():
            data[action] = self._get_stat_value(values, mode)
        return data

    def get_actions_durations_plot(self, return_plot=False):
        actions_durations = self.get_actions_durations(mode=StatMode.values)
        color_seeds = [name.split(".")[0] for name in actions_durations]
        return draw_stats(
            actions_durations,
            name="Actions durations",
            num=len(self),
            color_seeds=color_seeds,
            return_plot=return_plot,
        )

    @memoize
    def get_mean_duration(
        self, mode: str, only_successful: bool = True, only_single_picks: bool = False
    ) -> Optional[float]:
        if only_successful:
            items = self.successful_items
        if only_single_picks:
            items = [item for item in items if not item.multi_picks]
        total = len(items)
        if total == 0:
            return None
        duration_sum = 0
        if mode == StatDurations.total:
            total, duration_sum = self._get_total_duration(items)
        elif mode == StatDurations.aggregated:
            total, duration_sum = self._get_aggregated_duration(items)
        elif mode == StatDurations.end2end:
            duration_sum = self._get_end2end_duration(items)
        else:
            return
        if duration_sum is not None and total != 0:
            return duration_sum / total

    @exception_handler(dict())
    def get_mean_critical_path_categories_durations(self) -> Dict[str, float]:
        items = self.successful_items
        categories_durations = self._get_critical_path_categories_durations(items)
        for category, sum_duration in categories_durations.items():
            categories_durations[category] = sum_duration / len(items)
        return categories_durations

    @exception_handler
    def _get_total_duration(self, items: List[MongoDBItem]) -> Optional[float]:
        """
        total duration: time interval from:
         - start time of this item
        to:
         - end time of this item
        """
        total, duration_sum = 0, 0
        for item in items:
            duration = item.total_duration
            if duration is not None:
                total += 1
                duration_sum += duration
        return total, duration_sum

    @exception_handler
    def _get_aggregated_duration(self, items: List[MongoDBItem]) -> Optional[float]:
        """
        aggregated duration: sum of time that at least one action of this item takes place
        """
        total, duration_sum = 0, 0
        for item in items:
            duration = item.aggregated_duration
            if duration is not None:
                total += 1
                duration_sum += duration
        return total, duration_sum

    @exception_handler
    def _get_end2end_duration(self, items: List[MongoDBItem]) -> Optional[float]:
        """
        end2end duration: time interval from:
         - max(start time of this item, end time of previous item)
        to:
         - end time of this item
        """
        intervals = [(item.start_time, item.end_time) for item in items]
        return get_agregated_duration(intervals)

    def get_success_rate(self) -> float:
        if len(self) == 0:
            return -1.0
        return self.successful_items_num / len(self)

    @exception_handler(dict())
    def _get_critical_path_categories_durations(
        self, items: List[MongoDBItem]
    ) -> Dict[str, float]:
        sum_categories_durations = defaultdict(lambda: 0)
        for item in items:
            categories_durations = item.get_critical_path_categories_durations()
            if not categories_durations:
                continue
            for category, duration in categories_durations.items():
                sum_categories_durations[category] += duration
        return dict(sum_categories_durations)

    def get_fail_categories(self) -> Dict[str, List[MongoDBItem]]:
        fail_types = defaultdict(list)
        for item in self.items:
            error_category = item.error_category
            if error_category:
                fail_types[error_category].append(item)
        return fail_types

    def get_cls_methods(self) -> Dict[str, Set[str]]:
        cls_methods = defaultdict(set)
        for item in self.items:
            methods_logs = item.get_methods_logs()
            for method_log in methods_logs:
                cls_methods[method_log.cls_name].add(method_log.method_name)
        return dict(cls_methods)

    def get_method_durations(
        self,
        method_name: str,
        cls_name: Optional[str] = None,
        mode: StatMode = StatMode.mean,
    ) -> Dict[str, List[float]]:
        durations = []
        for item in self.items:
            methods_logs = item.get_methods_logs(
                cls_names=cls_name, method_names=method_name
            )
            durations.extend(
                [methods_log.duration for methods_log in methods_logs if methods_log]
            )
        result = self._get_stat_value(durations, mode)
        return result

    @staticmethod
    def get_stats(values):
        stats = {
            StatMode.mean: MongoDBStats._get_stat_value(values, StatMode.mean),
            StatMode.median: MongoDBStats._get_stat_value(values, StatMode.median),
            StatMode.stdev: MongoDBStats._get_stat_value(values, StatMode.stdev),
            StatMode.coef_var: MongoDBStats._get_stat_value(values, StatMode.coef_var),
            StatMode.count: MongoDBStats._get_stat_value(values, StatMode.count),
        }
        return stats

    @staticmethod
    def _get_stat_value(values, mode: StatMode):
        if mode == StatMode.values:
            return values
        elif mode == StatMode.count:
            return len(values)
        if len(values) == 0:
            return
        elif mode == StatMode.mean:
            return statistics.mean(values)
        elif mode == StatMode.median:
            return statistics.median(values)
        elif mode == StatMode.stdev:
            return statistics.stdev(values) if len(values) > 1 else None
        elif mode == StatMode.coef_var:
            if len(values) > 1:
                return statistics.stdev(values) / statistics.mean(values)
            else:
                return None
        else:
            return

    def get_summary(self, include_oids: bool = False) -> str:
        fail_categories = self.get_fail_categories()
        prev_fail_num = len(fail_categories.get(ECNames.previous_failure, []))
        items_num = len(self.items)
        items_num_pr = items_num - prev_fail_num
        succ_num = self.successful_items_num
        fail_num = self.failed_items_num
        fail_num_pr = self.failed_items_num - prev_fail_num
        summary = "Number of items" + " " * 12 + "(w/o previous failures)"
        summary += f"\n  Total items: {items_num:>15} ({items_num_pr})"
        if len(self.items) == 0:
            return summary

        start_time = str(self.start_time).split(".")[0] if self.start_time else None
        end_time = str(self.end_time).split(".")[0] if self.end_time else None
        succ_perc = round(succ_num / items_num * 100)
        succ_perc_pr = round(succ_num / items_num_pr * 100)
        fail_perc = round(fail_num / items_num * 100)
        fail_perc_pr = round(fail_num_pr / items_num_pr * 100)
        multi_picks_perc = round(self.multi_picks_items_num / items_num * 100)
        multi_picks_perc_pr = round(self.multi_picks_items_num / items_num_pr * 100)

        total_time = self.get_mean_duration(StatDurations.total)
        aggregated_time = self.get_mean_duration(StatDurations.aggregated)
        end2end_time = self.get_mean_duration(StatDurations.end2end)

        cats_durs = self.get_mean_critical_path_categories_durations()
        if cats_durs:
            cats_durs_perc = {
                k: (v, v / sum(cats_durs.values())) for k, v in cats_durs.items()
            }
            max_cats_durs_len = max([len(name) for name in cats_durs_perc])
        else:
            cats_durs_perc = None

        if multi_picks_perc > 0:
            single_pick_total_time = self.get_mean_duration(
                StatDurations.total, only_single_picks=True
            )
            single_pick_aggregated_time = self.get_mean_duration(
                StatDurations.aggregated, only_single_picks=True
            )
            single_pick_end2end_time = self.get_mean_duration(
                StatDurations.end2end, only_single_picks=True
            )

        max_cat_name_len = max([len(name) for name in get_cls_attributes(ECNames)])

        summary += (
            f"\n  Successful items: {self.successful_items_num:>10}"
            f" {succ_perc:>2}% ({succ_perc_pr:>2}%)"
        )
        summary += (
            f"\n  Failed items: {fail_num:>14} {fail_perc}% "
            f"({self.failed_items_num-prev_fail_num} {fail_perc_pr}%)"
        )
        summary += (
            f"\n  Multiple pick attempts: {self.multi_picks_items_num:>4}"
            f" {multi_picks_perc:>2}% ({multi_picks_perc_pr:>2}%)"
        )

        # --- Start/End times
        summary += f"\n\nStart time: {start_time}"
        summary += f"\nEnd time: {end_time}"

        # --- Durations
        if any((total_time, aggregated_time, end2end_time)):
            summary += f"\n\nMean Durations"
            if multi_picks_perc == 0:
                summary += " (Single Pick)"
        if total_time:
            summary += f"\n  total: {total_time:>13.3f} secs"
        if aggregated_time:
            summary += f"\n  aggregated: {aggregated_time:>8.3f} secs"
        if end2end_time:
            summary += f"\n  end2end: {end2end_time:>11.3f} secs"

        # --- Durations (Single Pick)
        if multi_picks_perc > 0:
            summary += f"\n Single Pick"
            if single_pick_total_time:
                summary += f"\n  total: {single_pick_total_time:>13.3f} secs"
            if single_pick_aggregated_time:
                summary += f"\n  aggregated: {single_pick_aggregated_time:>8.3f} secs"
            if single_pick_end2end_time:
                summary += f"\n  end2end: {single_pick_end2end_time:>11.3f} secs"

        # --- Categories Durations
        if cats_durs_perc:
            summary += f"\n\nCategories Durations ({sum([v[0] for v in cats_durs_perc.values()]):.3f} secs)"
            for cat, (dur, perc) in cats_durs_perc.items():
                summary += f"\n  {cat:<{max_cats_durs_len}}: {dur:.3f} secs ({(int(perc*100))}%)"

        summary += f"\n\nFail categories"
        for cat_name in sorted(get_cls_attributes(ECNames)):
            items = fail_categories.get(cat_name, [])
            cat_num = len(items)
            summary += (
                f"\n  {cat_name.replace('_', ' '):<{max_cat_name_len}}: {cat_num:>2}"
            )
            if items:
                total_perc = round(cat_num / items_num * 100)
                total_perc_pr = round(cat_num / items_num_pr * 100)
                fail_perc = round(cat_num / fail_num * 100)
                fail_perc_pr = (
                    round(cat_num / fail_num_pr * 100)
                    if cat_name != ECNames.previous_failure
                    else "--"
                )
                summary += f", {total_perc:>2}% ({total_perc_pr:>2}%) of total"
                summary += f", {fail_perc:>2}% ({fail_perc_pr:>2}%) of failed"
                if include_oids:
                    summary += f"\n    item info:"
                    for item in items:
                        summary += f"\n      {item.order_id}, {item.jan}"

        summary += f"\n\nGit SHA(s):"
        for sha in self.git_shas:
            summary += f"\n  {sha}"

        if self.git_branches:
            summary += f"\n\nGit Branch(es):"
            for branch in self.git_branches:
                summary += f"\n  {branch}"

        return summary

    def _get_items_from_hash(self, hash_id):
        file_path = os.path.join(OID_MAPS_DIR, f"{hash_id}.json")
        with open(file_path) as f:
            oids_dict = json.load(f)
        oids = oids_dict[OidMapsKeys.order_ids]
        db_reader = MongoDBReader()
        items = db_reader.get_items(
            order_ids=oids,
            elem_names=[
                GKeys.order_id,
                GKeys.metadata,
                AcKeys.action_results,
                CNs.OrderManagerNode,
                METHODS_LOGS_KEY,
            ],
            max_num=len(oids),
        )
        return items
