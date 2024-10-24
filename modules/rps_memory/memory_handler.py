import pdb
from typing import TypeVar, Generic, Any, Dict, List, Optional, Tuple, Union
import os
import yaml
import math
from mongodb import MongoDBMemory
from mongodb.constants import MemoryKeys as DBKeys
from mongodb.utils.decorators import exception_handler
import numpy as np


class FeatureNames:
    size = "size"
    size_after_pick_comparison = "size_after_pick_comparison"
    size_x = "size_x"
    size_y = "size_y"
    size_z = "size_z"
    weight_product = "weight"
    pads = "pads"
    pads_info = "pads_info"
    force_default_placing = "force_default_placing"
    item_classification = "item_classification"

class FeatureKeys:
    weight_entry = "weight"
    max_keep = "max_keep"
    estimation_criteria = "estimation_criteria"
    stable_criteria = "stable_criteria"
    min_num = "min_num"
    max_cv = "max_cv"
    min_successful_pick_ratio = "min_successful_pick_ratio"


FEATURES_CFG_PATH = os.path.join(os.path.dirname(__file__), "features_cfg.yaml")

SizeType = Tuple[float, float, float]
WeightType = float
T = TypeVar("T")


class MemoryHandler(Generic[T]):
    def __init__(self, features_cfg_path: str = FEATURES_CFG_PATH, *args, **kwargs):
        self._db_memory = MongoDBMemory(*args, **kwargs)
        self._configure(features_cfg_path)

    @exception_handler((False, None))
    def get_stable(self, feature_name: str, jan: str) -> Tuple[bool, Optional[T]]:
        feature = self._db_memory.get_feature(feature_name, jan=jan)
        if feature is None:
            return False, None
        # handle legacy (default)
        stable = (
            feature[DBKeys.stable]
            if DBKeys.stable in feature
            else feature.get(DBKeys.default)
        )
        if stable is not None:
            return True, stable
        return False, None

    @exception_handler(False)
    def set_stable(self, value: T, feature_name: str, jan: str) -> bool:
        return self._db_memory.set_stable(value, feature_name, jan=jan)

    @exception_handler
    def get_entries(
        self,
        feature_name: str,
        jan: str,
        as_dict: bool = False,
        flatten: bool = True,
    ):
        feature_data = self._db_memory.get_feature(feature_name, jan=jan)
        if feature_data is None:
            return None
        entries = feature_data[DBKeys.entries]
        if as_dict:
            return entries
        entries = list(entries.values())
        if flatten:
            entries = [entry for entry_list in entries for entry in entry_list]
        return entries

    @exception_handler
    def get_values(
        self,
        feature_name: str,
        jan: str,
        only_successful: bool = False,
    ) -> Optional[List[T]]:
        entries = self.get_entries(feature_name, jan, flatten=True)
        if entries is None:
            return None
        values = []
        for entry in entries:
            value = entry[DBKeys.value]
            if only_successful:

                if DBKeys.successful in entry and entry[DBKeys.successful]:
                    values.append(value)
            else:
                values.append(value)
        return values

    @exception_handler(False)
    def insert_values(
        self,
        feature_name: str,
        jan: str,
        values: List[T],
        order_id: str,
        is_successful: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        has_np: bool = False,
        max_keep: Optional[int] = None,
    ) -> bool:
        ret = True
        for value in values:
            ret &= self._db_memory.insert_entry(
                feature_name=feature_name,
                jan=jan,
                value=value,
                order_id=order_id,
                successful=is_successful,
                has_np=has_np,
                metadata=metadata,
            )
            if not ret:
                return ret
        entries = self.get_entries(feature_name=feature_name, jan=jan, flatten=True)
        if max_keep is not None and max_keep > 0 and len(entries) > max_keep:
            self._db_memory.pop_entries(
                feature_name, jan=jan, num=len(entries) - max_keep
            )
        return True

    @exception_handler(False)
    def insert_value(
        self,
        feature_name: str,
        jan: str,
        value: T,
        order_id: str,
        is_successful: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        has_np: bool = False,
        max_keep: Optional[int] = None,
    ) -> bool:
        return self.insert_values(
            feature_name=feature_name,
            jan=jan,
            values=[value],
            order_id=order_id,
            is_successful=is_successful,
            metadata=metadata,
            has_np=has_np,
            max_keep=max_keep,
        )

    @exception_handler(False)
    def remove_entry(
        self,
        feature_name: str,
        jan: str,
        order_id: str,
    ) -> bool:
        return self._db_memory.remove_entry(
            feature_name=feature_name, jan=jan, order_id=order_id
        )

    @exception_handler(False)
    def remove_elem(
        self,
        feature_name: str,
        jan: str,
        order_id: str,
        index: int,
    ) -> bool:
        return self._db_memory.remove_elem(
            feature_name=feature_name,
            jan=jan,
            order_id=order_id,
            index=index,
        )

    @exception_handler(False)
    def update_entry(
        self,
        feature_name: str,
        jan: str,
        update_dict: Dict[str, Any],
        order_id: str,
    ) -> bool:
        return self._db_memory.update_entry(
            feature_name=feature_name,
            jan=jan,
            update_dict=update_dict,
            order_id=order_id,
        )

    @exception_handler(False)
    def update_elem(
        self,
        feature_name: str,
        jan: str,
        update_dict: Dict[str, Any],
        order_id: str,
        index: int,
    ) -> bool:
        return self._db_memory.update_elem(
            feature_name=feature_name,
            jan=jan,
            update_dict=update_dict,
            order_id=order_id,
            index=index,
        )

        # @exception_handler((False, None))
        # def get_estimation(
        #     self,
        #     feature_name: str,
        #     jan: str,
        #     measured_value: Optional[T] = None,
        #     order_id: Optional[str] = None,
        #     update_feature_flag: Optional[bool] = True,
        # ) -> Tuple[bool, Optional[T]]:
        #     # rec, stable = self.get_stable(feature_name, jan=jan)
        #     # if rec:
        #     #     return True, stable
        #     feature_data = self._db_memory.get_feature(feature_name, jan=jan)
        #     values, stats = self._get_values_stats(feature_data)
        #     use_estimation = self._check_criteria(
        #         feature_name,
        #         FeatureKeys.estimation_criteria,
        #         stats,
        #         jan,
        #     )
        #     if not use_estimation:
        #         if measured_value is None:
        #             return False, None
        #         rec, estimation = True, measured_value
        #     elif feature_name == FeatureNames.size:
        #         rec, estimation = self._get_size_estimation(
        #             measured_value, values, stats, jan
        #         )
        #     elif feature_name == FeatureNames.weight_product:
        #         rec, estimation = self._get_weight_estimation(
        #             measured_value, values, stats, jan
        #         )
        #     elif feature_name == FeatureNames.pattern_id:
        #         rec, estimation = self._get_pattern_id_estimation(
        #             measured_value, values, stats, jan
        #         )


        #     if measured_value is not None and update_feature_flag:
        #         self._update_feature(
        #             measured_value,
        #             feature_data,
        #             feature_name=feature_name,
        #             jan=jan,
        #             order_id=order_id,
        #         )
        #     return rec, estimation

        # @exception_handler((list(), dict()))
        # def _get_values_stats(
        #     self, feature_data: Optional[Dict[str, Any]]
        # ) -> Tuple[List, Dict]:
        #     if feature_data is None:
        #         return list(), dict()
        #     entries = feature_data[DBKeys.entries]
        #     values = []
        #     for v_list in entries.values():
        #         values.extend([v[DBKeys.value] for v in v_list if v[DBKeys.successful]])
        #     stats = feature_data[DBKeys.stats]
        #     return values, stats

        # @exception_handler
        # def _update_feature(
        #     self,
        #     new_value: T,
        #     feature_data: Optional[Dict[str, Any]],
        #     feature_name: str,
        #     jan: str,
        #     order_id: Optional[str] = None,
        # ) -> None:
        #     if new_value is None:
        #         return
        #     self._update_entries(
        #         new_value=new_value,
        #         feature_data=feature_data,
        #         feature_name=feature_name,
        #         jan=jan,
        #         order_id=order_id,
        #     )
        #     stats = self._update_stats(
        #         new_value=new_value,
        #         feature_data=feature_data,
        #         feature_name=feature_name,
        #         jan=jan,
        #     )
        #     self._update_stability(stats, feature_name=feature_name, jan=jan)

        # @exception_handler
        # def _update_entries(
        #     self,
        #     new_value: T,
        #     feature_data: Optional[Dict[str, Any]],
        #     feature_name: str,
        #     jan: str,
        #     order_id: Optional[str] = None,
        # ):
        #     if new_value is None:
        #         return
        #     self._db_memory.insert_entry(
        #         feature_name, new_value, jan=jan, order_id=order_id
        #     )
        #     values, _ = self._get_values_stats(feature_data)
        #     if values:
        #         max_keep = self._get_feature_field(feature_name, FeatureKeys.max_keep, jan)
        #         for _ in range(max(0, len(values) + 1 - max_keep)):
        #             self._db_memory.pop_entries(feature_name, jan=jan)

        # @exception_handler
        # def _update_stats(
        #     self,
        #     new_value: T,
        #     feature_data: Optional[Dict[str, Any]],
        #     feature_name: str,
        #     jan: str,
        #     weight: float = 1,
        # ) -> Optional[Dict[str, Any]]:
        #     if new_value is None:
        #         return
        #     _, stats = self._get_values_stats(feature_data)
        #     if not stats:
        #         stats = self._get_new_stats(new_value, feature_name)
        #     else:
        #         stats = self._get_updated_stats(new_value, stats, feature_name, weight)
        #     if stats is not None:
        #         self._db_memory.set_stats(stats, feature_name, jan=jan)
        #         return stats

        # @exception_handler
        # def _update_stability(self, stats: Optional[Dict], feature_name: str, jan: str):
        #     is_stable = self._check_criteria(
        #         feature_name, FeatureKeys.stable_criteria, stats, jan
        #     )
        #     if not is_stable:
        #         return
        #     stable = stats[DBKeys.avg]
        #     # self.set_stable(stable, feature_name=feature_name, jan=jan)

        # @exception_handler
        # def _get_new_stats(self, new_value: T, feature_name: str) -> Optional[Dict]:
        #     stats = {
        #         DBKeys.avg: new_value,
        #         DBKeys.counter: 1,
        #     }
        #     if feature_name == FeatureNames.size:
        #         stats[DBKeys.stdev] = (0, 0, 0)
        #         stats[DBKeys.cv] = (0, 0, 0)
        #     else:
        #         stats[DBKeys.stdev] = 0
        #         stats[DBKeys.cv] = 0
        #     return stats

        # @exception_handler((False, None))
        # def _get_size_estimation(
        #     self,
        #     measured_value: SizeType,
        #     values: List[SizeType],
        #     stats: Optional[Dict],
        #     jan: str,
        # ) -> Tuple[bool, Optional[SizeType]]:
        #     values_np = np.asarray(values)
        #     estimation_median = np.median(values_np, axis=0)
        #     estimation_size = estimation_median

        #     measured_value_np = np.array(measured_value)

        #     if measured_value is None or len(values) < 20:
        #         # If the memory and measurement differ a lot, be on the safe side and use maximum of every dimension.
        #         if measured_value is not None:
        #             size_diff_measured = np.abs(estimation_size - measured_value_np)
        #             if np.max(size_diff_measured) > 20:
        #                 estimation_size = np.maximum(estimation_size, measured_value_np)

        #         return True, estimation_size.tolist()

        #     values_np_norm = StandardScaler().fit_transform(values_np)
        #     db = DBSCAN(eps=0.3, min_samples=10).fit(values_np_norm)
        #     labels = db.labels_
        #     unique_labels = set(labels)
        #     dist_cluster_mean = 10000
        #     max_member_cluster = 0
        #     stable_value = estimation_median
        #     # Go over the clusters and find the most similar cluster center to the measure size (input).
        #     # Also, assign the cluster center of the most crowded cluster as the stable value.
        #     for k in unique_labels:
        #         if k == -1:
        #             continue
        #         class_member_mask = labels == k
        #         no_of_cluster_member = np.sum(class_member_mask)
        #         if no_of_cluster_member < 5:
        #             continue
        #         values_class = values_np[class_member_mask, :]
        #         cluster_mean = np.mean(values_class, axis=0)
        #         dist_cluster_mean_iter = np.sum(abs(cluster_mean - measured_value_np))
        #         if dist_cluster_mean_iter < dist_cluster_mean:
        #             dist_cluster_mean = dist_cluster_mean_iter
        #             estimation_size = cluster_mean
        #         if no_of_cluster_member > max_member_cluster:
        #             max_member_cluster = no_of_cluster_member
        #             stable_value = cluster_mean

        #     # If the memory and measurement differ a lot, be on the safe side and use maximum of every dimension.
        #     size_diff_measured = np.abs(estimation_size - measured_value_np)
        #     if np.max(size_diff_measured) > 20:
        #         estimation_size = np.maximum(estimation_size, measured_value_np)

        #     # Set the stable value to the most promising cluster center.
        #     self.set_stable(stable_value.tolist(), FeatureNames.size, jan)

        return True, estimation_size.tolist()


    @exception_handler
    def _get_feature_field(
        self,
        feature_name: str,
        field_name: str,
        jan: Optional[str] = None,
        default=None,
    ):
        try:
            return self.features_cfg[jan][feature_name][field_name]
        except:
            try:
                return self.features_cfg["default"][feature_name][field_name]
            except:
                pass
        return default

    @exception_handler(False)
    def _configure(self, config_file: str):
        try:
            with open(config_file) as f:
                self.features_cfg = yaml.safe_load(f)
            return True
        except:
            self.features_cfg = dict()
            return False

    @exception_handler(False)
    def _check_criteria(
        self,
        feature_name: str,
        criteria_name: str,
        stats: Optional[Dict],
        jan: str,
    ) -> bool:
        if not stats:
            return False
        criteria = self._get_feature_field(feature_name, criteria_name, jan)
        if FeatureKeys.min_num in criteria:
            if stats[DBKeys.counter] < criteria[FeatureKeys.min_num]:
                return False
        if FeatureKeys.min_successful_pick_ratio in criteria:
            if (stats[DBKeys.counter_successful] / stats[DBKeys.counter]) < criteria[
                FeatureKeys.min_successful_pick_ratio
            ]:
                return False
        # if FeatureKeys.max_cv in criteria:
        #     if stats[DBKeys.cv] > criteria[FeatureKeys.max_cv]:
        #         return False
        return True

    @staticmethod
    @exception_handler
    def _get_updated_stats(
        new_val: Union[float, List[float]],
        stats: Dict,
        feature_name: str,
        w: float = 1.0,
    ) -> Optional[Dict]:
        old_avg = stats.get(DBKeys.avg)
        old_stdev = stats.get(DBKeys.stdev)
        counter = stats.get(DBKeys.counter)

        # Nested function for updating a single statistic
        def update_single_stat(counter, avg, std_dev, val, weight):
            new_avg = (counter * avg + weight * val) / (counter + weight)
            old_ssd = (counter - 1) * std_dev**2
            new_ssd = old_ssd + weight * (val - new_avg) * (val - avg)
            new_stdev = math.sqrt(new_ssd / (counter + weight - 1))
            new_coef_var = new_stdev / new_avg if new_avg != 0 else float("inf")
            return new_avg, new_stdev, new_coef_var

        if isinstance(new_val, (tuple, list)):
            # Ensure all inputs are tuples of the same length
            if not isinstance(old_avg, (tuple, list)):
                old_avg = (old_avg,) * len(new_val)
            if not isinstance(old_stdev, (tuple, list)):
                old_stdev = (old_stdev,) * len(new_val)
            if not isinstance(w, (tuple, list)):
                w = (w,) * len(new_val)

            # Update each statistic and store the results in lists
            new_avgs = []
            new_stdevs = []
            new_coef_vars = []
            for val, mean, std_dev, weight in zip(new_val, old_avg, old_stdev, w):
                new_avg, new_stdev, new_coef_var = update_single_stat(
                    counter, mean, std_dev, val, weight
                )
                new_avgs.append(new_avg)
                new_stdevs.append(new_stdev)
                new_coef_vars.append(new_coef_var)

            new_stats = {
                DBKeys.avg: tuple(new_avgs),
                DBKeys.stdev: tuple(new_stdevs),
                DBKeys.cv: tuple(new_coef_vars),
                DBKeys.counter: counter + 1,
            }
        else:
            # Inputs are not tuples, handle as single statistics
            new_avg, new_stdev, new_coef_var = update_single_stat(
                counter, old_avg, old_stdev, new_val, w
            )

            new_stats = {
                DBKeys.avg: new_avg,
                DBKeys.stdev: new_stdev,
                DBKeys.cv: new_coef_var,
                DBKeys.counter: counter + 1,
            }

        return new_stats

    @staticmethod
    @exception_handler
    def _get_updated_stats_pads_info(
        new_val: Dict, stats: Dict, successful: Optional[bool] = True
    ) -> Optional[Dict]:
        counter = stats.get(DBKeys.counter)
        counter_successful = stats.get(DBKeys.counter_successful)

        new_stats = {
            DBKeys.counter: counter + 1,
            DBKeys.counter_successful: counter_successful + int(successful),
        }
        return new_stats
