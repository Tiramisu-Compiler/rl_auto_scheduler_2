from copy import deepcopy
from typing import Any, Dict, Tuple

from env_api.utils.data_preprocessors import (
    construct_tree_structure,
    get_ancestory_register,
)


class Action:
    def __init__(
        self,
        params: list,
        name: str,
        comps: list = [],
        env_id: int = None,
        worker_id="",
    ):
        self.params = params
        self.name = name
        # List of computations concerned by this action
        self.comps = comps
        # The ID of the action inside the RL system, this ID helps in masking actions and matching RL with env_api
        self.env_id = env_id
        # In distributed training we want to know which worker is applying this action in order to distinguish the compilation
        # of the same function by different workers
        self.worker_id = worker_id

    def __str__(self) -> str:
        return f"\n{self.name}(params={self.params}, comps={self.comps})"

    def __repr__(self) -> str:
        return f"\n{self.name}(params={self.params}, comps={self.comps})"


class AffineAction(Action):
    def __init__(
        self,
        params: list,
        name: str,
        comps: list = [],
        env_id: int = None,
        worker_id="",
    ):
        super().__init__(params, name, comps, env_id, worker_id)


class Reversal(AffineAction):
    def __init__(self, params: list, env_id: int = None, worker_id=""):
        super().__init__(params, name="Reversal", env_id=env_id, worker_id=worker_id)


class Interchange(AffineAction):
    def __init__(self, params: list, env_id: int = None, worker_id=""):
        super().__init__(params, name="Interchange", env_id=env_id, worker_id=worker_id)


class Skewing(AffineAction):
    def __init__(self, params: list, env_id: int = None, worker_id=""):
        super().__init__(params, name="Skewing", env_id=env_id, worker_id=worker_id)


class Parallelization(Action):
    def __init__(self, params: list, env_id: int = None, worker_id=""):
        super().__init__(
            params, name="Parallelization", env_id=env_id, worker_id=worker_id
        )


class Unrolling(Action):
    def __init__(self, params: list, env_id: int = None, worker_id=""):
        super().__init__(params, name="Unrolling", env_id=env_id, worker_id=worker_id)


class Tiling(Action):
    def __init__(self, params: list, env_id: int = None, worker_id=""):
        super().__init__(params, name="Tiling", env_id=env_id, worker_id=worker_id)
        self.subtilings = []

    def __str__(self) -> str:
        size = len(self.params) // 2

        return (
            f"\nTiling {size}D : on levels "
            + ",".join([str(x) for x in self.params[:size]])
            + " on comps "
            + ",".join(self.comps)
            + "".join([str(tiling) for tiling in self.subtilings])
        )


class Fusion(Action):
    def __init__(self, params: list, env_id: int = None, worker_id=""):
        super().__init__(params, name="Fusion", env_id=env_id, worker_id=worker_id)

    @staticmethod
    def get_tree_structure_after_fusion(
        fusion_candidates: Tuple[str, str], program_annotations: Dict[str, Any]
    ):
        """
        Construct the tree structure of the program then fuse the two computations in the tree structure

        Args:
            fusion_candidates (Tuple[str, str]): The two computations to be fused
            program_annotations (Dict[str, Any]): The program annotations

        Returns:
            Dict[str, Any]: The tree structure of the program after fusing the two computations
        """
        first_comp, second_comp = fusion_candidates
        first_comp_dict = program_annotations["computations"][first_comp]
        second_comp_dict = program_annotations["computations"][second_comp]

        # The computations to be fused should be consecutive (following the c++ autischeduler)
        assert (
            first_comp_dict["absolute_order"] + 1 == second_comp_dict["absolute_order"]
        ), f"The two computations to be fused are not consecutive ({first_comp_dict['absolute_order']}, {second_comp_dict['absolute_order']})"

        iterator_first_comp = first_comp_dict["iterators"][-1]
        iterator_second_comp = second_comp_dict["iterators"][-1]

        # The computations to be fused should have different parent iterators otherwise they are already fused
        assert (
            iterator_first_comp != iterator_second_comp
        ), "The two computations to be fused have the same parent iterator"

        # Construct the tree structure of the initial program
        tree_structure = construct_tree_structure(program_annotations)

        # get ancestory register of first_comp
        first_comp_ancestory = get_ancestory_register(first_comp_dict, tree_structure)
        second_comp_ancestory = get_ancestory_register(second_comp_dict, tree_structure)

        first_comp_parent_dict = first_comp_ancestory[iterator_first_comp]
        second_comp_parent_dict = second_comp_ancestory[iterator_second_comp]

        # copy the child iterators and computations of the second computation iterator to the first computation
        first_comp_parent_dict["child_list"].extend(
            second_comp_parent_dict["child_list"]
        )
        first_comp_parent_dict["computations_list"].extend(
            second_comp_parent_dict["computations_list"]
        )

        # emptry the child iterators and computations of the second computation iterator
        second_comp_parent_dict["child_list"] = []
        second_comp_parent_dict["computations_list"] = []

        # Remove empty iterators (both computations and iterators) from the tree structure
        for idx, ancestor in reversed(list(enumerate(second_comp_dict["iterators"]))):
            ancestor_dict = second_comp_ancestory[ancestor]
            # if the iterator has no child iterators and no computations
            if (
                len(ancestor_dict["child_list"]) == 0
                and len(ancestor_dict["computations_list"]) == 0
            ):
                # remove the iterator from its parent iterators
                if idx - 1 >= 0:
                    parent_dict = second_comp_ancestory[
                        second_comp_dict["iterators"][idx - 1]
                    ]
                    parent_dict["child_list"].remove(ancestor_dict)
                else:
                    # if the iterator is a root iterator then remove it from the roots list
                    tree_structure["roots"].remove(ancestor_dict)
            else:
                break

        return tree_structure

    @staticmethod
    def fuse_annotations(
        fusion_candidates: Tuple[str, str], program_annotations_dict: Dict[str, Any]
    ):
        """
        Fuses the two computations in the program annotations dictionary

        Args:
            fusion_candidates (Tuple[str, str]): The two computations to be fused
            program_annotations_dict (Dict[str, Any]): The program annotations

        Returns:
            Dict[str, Any]: The program annotations after fusing the two computations
        """
        # Make a deep copy of the program annotations
        program_annotations = deepcopy(program_annotations_dict)
        first_comp, second_comp = fusion_candidates
        first_comp_dict = program_annotations["computations"][first_comp]
        second_comp_dict = program_annotations["computations"][second_comp]

        assert (
            first_comp_dict["absolute_order"] + 1 == second_comp_dict["absolute_order"]
        ), f"The two computations to be fused are not consecutive ({first_comp_dict['absolute_order']}, {second_comp_dict['absolute_order']})"

        iterator_first_comp = first_comp_dict["iterators"][-1]
        iterator_second_comp = second_comp_dict["iterators"][-1]

        assert (
            iterator_first_comp != iterator_second_comp
        ), "The two computations to be fused have the same parent iterator"

        # Get the iterators of the computations to be fused
        iterator_first_comp_dict = program_annotations["iterators"][iterator_first_comp]
        iterator_second_comp_dict = program_annotations["iterators"][
            iterator_second_comp
        ]

        # copy the child iterators and computations of the second computation iterator to the first computation
        iterator_first_comp_dict["child_iterators"].extend(
            iterator_second_comp_dict["child_iterators"]
        )
        iterator_first_comp_dict["computations_list"].extend(
            iterator_second_comp_dict["computations_list"]
        )

        # change the parent iterator of all the sencode computation's iterators to the first computation's iterator
        for child_iterator in iterator_second_comp_dict["child_iterators"]:
            child_iterator_dict = program_annotations["iterators"][child_iterator]
            child_iterator_dict["parent_iterator"] = iterator_first_comp
            # change the ancestory of the child iterator's computations to the first computation's ancestory
            for computation in child_iterator_dict["computations_list"]:
                computation_dict = program_annotations["computations"][computation]
                computation_dict["iterators"] = first_comp_dict["iterators"] + [
                    child_iterator
                ]

        # change the ancestory of the second computation's computations to the first computation's ancestory
        for computation in iterator_second_comp_dict["computations_list"]:
            computation_dict = program_annotations["computations"][computation]
            computation_dict["iterators"] = first_comp_dict["iterators"]

        # empty the child iterators and computations of the second computation iterator
        iterator_second_comp_dict["child_iterators"] = []
        iterator_second_comp_dict["computations_list"] = []

        # Loop through the ancestors of the second computation's iterator and remove all empty iterators
        current_iterator = iterator_second_comp
        while current_iterator is not None:
            current_iterator_dict = program_annotations["iterators"][current_iterator]
            if (
                len(current_iterator_dict["child_iterators"]) == 0
                and len(current_iterator_dict["computations_list"]) == 0
            ):
                parent_iterator = current_iterator_dict["parent_iterator"]
                if parent_iterator is not None:
                    parent_iterator_dict = program_annotations["iterators"][
                        parent_iterator
                    ]
                    parent_iterator_dict["child_iterators"].remove(current_iterator)
                program_annotations["iterators"].pop(current_iterator)
                current_iterator = parent_iterator
            else:
                break

        return program_annotations
