import copy
import json
import re

import numpy as np
import torch

from env_api.scheduler.models.action import Fusion
from env_api.utils.exceptions import *

# Maximum sequence of transformations (reversal, interchange and skewing) allowed. Currently set to 4
MAX_NUM_TRANSFORMATIONS = 4

# Maximum size of the tags vector representing each transformation
MAX_TAGS = 16

MAX_DEPTH = 5


class ConvertService:
    @classmethod
    def get_representation_template(
        cls, program_json, no_sched_json, max_depth=MAX_DEPTH, train_device="cpu"
    ):
        # Set the max and min number of accesses allowed
        max_accesses = 15
        min_accesses = 0
        comps_repr_templates_list = []
        comps_expr_repr_templates_list = []
        comps_indices_dict = dict()
        comps_placeholders_indices_dict = dict()

        # Get the computations (program statements) dictionary and order them according to the absolute_order attribute
        computations_dict = program_json["computations"]
        ordered_comp_list = sorted(
            list(computations_dict.keys()),
            key=lambda x: computations_dict[x]["absolute_order"],
        )

        for comp_index, comp_name in enumerate(ordered_comp_list):
            comp_dict = computations_dict[comp_name]
            expr_dict = comp_dict["expression_representation"]
            comp_type = comp_dict["data_type"]
            comps_expr_repr_templates_list.append(
                cls.get_tree_expr_repr(expr_dict, comp_type)
            )
            if len(comp_dict["accesses"]) > max_accesses:
                raise NbAccessException

            if len(comp_dict["accesses"]) < min_accesses:
                raise NbAccessException

            if len(comp_dict["iterators"]) > max_depth:
                raise LoopsDepthException

            comp_repr_template = []

            comp_repr_template.append(+comp_dict["comp_is_reduction"])

            iterators_repr = []
            # Add a representation of each loop of this computation
            for iter_i, iterator_name in enumerate(comp_dict["iterators"]):
                # TODOF does this work when iterators have the same name?
                iterator_dict = program_json["iterators"][iterator_name]
                # Create a unique code for each loop
                c_code = "C" + str(comp_index)
                l_code = c_code + "-L" + str(iter_i)

                # Add a placeholder for transformations applied to this loop
                iterators_repr.extend(
                    [
                        l_code + "Parallelized",
                        l_code + "Tiled",
                        l_code + "TileFactor",
                        l_code + "Fused",
                        l_code + "Shifted",
                        l_code + "ShiftFactor",
                    ]
                )

            iterator_repr_size = int(len(iterators_repr) / len(comp_dict["iterators"]))

            # Add padding incase the number of loops is lower than the max
            iterators_repr.extend(
                [0] * iterator_repr_size * (max_depth - len(comp_dict["iterators"]))
            )

            # Add two tags for whether unrolling was applied and the unrolling factor
            iterators_repr.extend([c_code + "-Unrolled", c_code + "-UnrollFactor"])

            # Add a placeholder for the other transformations to be applied (skewing, reversal and interchage)
            iterators_repr.append(c_code + "-TransformationTagsStart")
            iterators_repr.extend(["M"] * (MAX_TAGS * MAX_NUM_TRANSFORMATIONS - 2))
            iterators_repr.append(c_code + "-TransformationTagsEnd")

            # Adding initial constraint matrix
            iterators_repr.append(c_code + "-OgConstraintMatrixStart")
            iterators_repr.extend(["C"] * ((max_depth + 1) * ((max_depth + 1) * 2) - 2))
            iterators_repr.append(c_code + "-OgConstraintMatrixEnd")

            # Adding initial constraint matrix
            iterators_repr.append(c_code + "-OgConstraintVectorStart")
            iterators_repr.extend(["V"] * (max_depth * 2 - 2))
            iterators_repr.append(c_code + "-OgConstraintVectorEnd")

            # Adding transformed constraint matrix
            iterators_repr.append(c_code + "-ConstraintMatrixStart")
            iterators_repr.extend(["C"] * ((max_depth + 1) * ((max_depth + 1) * 2) - 2))
            iterators_repr.append(c_code + "-ConstraintMatrixEnd")

            # Add the loop representation to the computation vector
            comp_repr_template.extend(iterators_repr)

            # Pad the write access matrix and add it to the representation
            padded_write_matrix = cls.pad_access_matrix(
                cls.isl_to_write_matrix(comp_dict["write_access_relation"]), max_depth
            )
            write_access_repr = [
                comp_dict["write_buffer_id"] + 1
            ] + padded_write_matrix.flatten().tolist()

            comp_repr_template.extend(write_access_repr)

            # Pad the read access matrix and add it to the representation
            # Todo add details about the read accesses
            read_accesses_repr = []
            for read_access_dict in comp_dict["accesses"]:
                read_access_matrix = cls.pad_access_matrix(
                    read_access_dict["access_matrix"], max_depth
                )
                read_access_repr = (
                    [+read_access_dict["access_is_reduction"]]
                    + [read_access_dict["buffer_id"] + 1]
                    + read_access_matrix.flatten().tolist()
                )
                read_accesses_repr.extend(read_access_repr)

            access_repr_len = (max_depth + 1) * (max_depth + 2) + 1 + 1

            read_accesses_repr.extend(
                [0] * access_repr_len * (max_accesses - len(comp_dict["accesses"]))
            )

            comp_repr_template.extend(read_accesses_repr)

            comps_repr_templates_list.append(comp_repr_template)

            # Create a mapping between the features and their position in the representation
            comps_indices_dict[comp_name] = comp_index
            for j, element in enumerate(comp_repr_template):
                if isinstance(element, str):
                    comps_placeholders_indices_dict[element] = (comp_index, j)

        # Create a representation of the loops independantly from the computations
        loops_repr_templates_list = []
        loops_indices_dict = dict()
        loops_placeholders_indices_dict = dict()

        for loop_index, loop_name in enumerate(program_json["iterators"]):
            # Create a unique code for each loop
            loop_repr_template = []
            l_code = "L" + loop_name

            # Add a placeholder for transformations applied to this loop
            loop_repr_template.extend(
                [
                    l_code + "Parallelized",
                    l_code + "Tiled",
                    l_code + "TileFactor",
                    l_code + "Fused",
                    l_code + "Unrolled",
                    l_code + "UnrollFactor",
                    l_code + "Shifted",
                    l_code + "ShiftFactor",
                ]
            )

            # Create a mapping between the features and their position in the representation
            loops_repr_templates_list.append(loop_repr_template)
            loops_indices_dict[loop_name] = loop_index

            for j, element in enumerate(loop_repr_template):
                if isinstance(element, str):
                    loops_placeholders_indices_dict[element] = (loop_index, j)

        # Make sure no fusion was applied on this version and get the original tree structure
        assert "fusions" not in no_sched_json or no_sched_json["fusions"] == None

        orig_tree_structure = no_sched_json["tree_structure"]
        tree_annotation = copy.deepcopy(orig_tree_structure)

        prog_tree = cls.update_tree_atributes(
            tree_annotation,
            loops_indices_dict,
            comps_indices_dict,
            train_device=train_device,
        )

        # adding padding to the expressions to get same size expressions
        max_exprs = 0
        max_exprs = max([len(comp) for comp in comps_expr_repr_templates_list])
        lengths = []
        for j in range(len(comps_expr_repr_templates_list)):
            lengths.append(len(comps_expr_repr_templates_list[j]))
            comps_expr_repr_templates_list[j].extend(
                [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
                * (max_exprs - len(comps_expr_repr_templates_list[j]))
            )
        comps_expr_lengths = torch.tensor(lengths)
        comps_expr_repr_templates_list = torch.tensor([comps_expr_repr_templates_list])
        return (
            prog_tree,
            comps_repr_templates_list,
            loops_repr_templates_list,
            comps_placeholders_indices_dict,
            loops_placeholders_indices_dict,
            comps_expr_repr_templates_list,
            comps_expr_lengths,
        )

    @classmethod
    def update_tree_atributes(
        cls, node, loops_indices_dict, comps_indices_dict, train_device="cpu"
    ):
        if "roots" in node:
            for root in node["roots"]:
                cls.update_tree_atributes(
                    root,
                    loops_indices_dict,
                    comps_indices_dict,
                    train_device=train_device,
                )
            return node
        node["loop_index"] = loops_indices_dict[node["loop_name"]]
        if node["computations_list"] != []:
            node["computations_indices"] = [
                comps_indices_dict[comp_name] for comp_name in node["computations_list"]
            ]

            node["has_comps"] = True
        else:
            node["has_comps"] = False
        for child_node in node["child_list"]:
            cls.update_tree_atributes(
                child_node,
                loops_indices_dict,
                comps_indices_dict,
                train_device=train_device,
            )
        return node

    @classmethod
    def get_tree_expr_repr(cls, node, comp_type):
        expr_tensor = []
        if node["children"] != []:
            for child_node in node["children"]:
                expr_tensor.extend(cls.get_tree_expr_repr(child_node, comp_type))
        expr_tensor.append(cls.get_expr_repr(node["expr_type"], comp_type))

        return expr_tensor

    @classmethod
    def get_expr_repr(cls, expr, comp_type):
        expr_vector = []
        if expr == "add":
            expr_vector = [1, 0, 0, 0, 0, 0, 0, 0]
        elif expr == "sub":
            expr_vector = [0, 1, 0, 0, 0, 0, 0, 0]
        elif expr == "mul":
            expr_vector = [0, 0, 1, 0, 0, 0, 0, 0]
        elif expr == "div":
            expr_vector = [0, 0, 0, 1, 0, 0, 0, 0]
        elif expr == "sqrt":
            expr_vector = [0, 0, 0, 0, 1, 0, 0, 0]
        elif expr == "min":
            expr_vector = [0, 0, 0, 0, 0, 1, 0, 0]
        elif expr == "max":
            expr_vector = [0, 0, 0, 1, 0, 0, 1, 0]
        else:
            expr_vector = [0, 0, 0, 1, 0, 0, 0, 1]

        comp_type_vector = []
        if comp_type == "int32":
            comp_type_vector = [1, 0, 0]
        elif comp_type == "float32":
            comp_type_vector = [0, 1, 0]
        elif comp_type == "float64":
            comp_type_vector = [0, 0, 1]

        return expr_vector + comp_type_vector

    @classmethod
    def pad_access_matrix(cls, access_matrix, max_depth):
        access_matrix = np.array(access_matrix)
        access_matrix = np.c_[np.ones(access_matrix.shape[0]), access_matrix]
        access_matrix = np.r_[[np.ones(access_matrix.shape[1])], access_matrix]
        padded_access_matrix = np.zeros((max_depth + 1, max_depth + 2))
        padded_access_matrix[
            : access_matrix.shape[0], : access_matrix.shape[1] - 1
        ] = access_matrix[:, :-1]
        padded_access_matrix[: access_matrix.shape[0], -1] = access_matrix[:, -1]

        return padded_access_matrix

    @classmethod
    def linear_diophantine_default(cls, f_i, f_j):
        found = False
        gamma = 0
        sigma = 1
        if (f_j == 1) or (f_i == 1):
            gamma = f_i - 1
            sigma = 1
        else:
            if (f_j == -1) and (f_i > 1):
                gamma = 1
                sigma = 0
            else:
                i = 0
                while (i < 100) and (not found):
                    if ((sigma * f_i) % abs(f_j)) == 1:
                        found = True
                    else:
                        sigma += 1
                        i += 1
                if not found:
                    print("Error cannof find solution to diophantine equation")
                    return
                gamma = ((sigma * f_i) - 1) / f_j

        return gamma, sigma

    @classmethod
    def isl_to_write_matrix(cls, isl_map):
        comp_iterators_str = re.findall(r"\[(.*)\]\s*->", isl_map)[0]
        buffer_iterators_str = re.findall(r"->\s*\w*\[(.*)\]", isl_map)[0]
        buffer_iterators_str = re.sub(r"\w+'\s=", "", buffer_iterators_str)
        comp_iter_names = re.findall(r"(?:\s*(\w+))+", comp_iterators_str)
        buf_iter_names = re.findall(r"(?:\s*(\w+))+", buffer_iterators_str)
        matrix = np.zeros([len(buf_iter_names), len(comp_iter_names) + 1])
        for i, buf_iter in enumerate(buf_iter_names):
            for j, comp_iter in enumerate(comp_iter_names):
                if buf_iter == comp_iter:
                    matrix[i, j] = 1
                    break
        return matrix

    @classmethod
    def isl_to_write_dims(cls, isl_map):
        buffer_iterators_str = re.findall(r"->\s*\w*\[(.*)\]", isl_map)[0]
        buffer_iterators_str = re.sub(r"\w+'\s=", "", buffer_iterators_str)
        buf_iter_names = re.findall(r"(?:\s*(\w+))+", buffer_iterators_str)
        return buf_iter_names

    @classmethod
    def get_static_dims(cls, schedule_json, program_json):
        tree_structure = schedule_json["tree_structure"]
        result = []
        nb_comps = len(program_json["computations"])

        for i in range(nb_comps):
            result.append([])

        level = tree_structure
        to_explore = [root for root in tree_structure["roots"]]
        # to_explore.append(tree_structure)
        while to_explore:
            level = to_explore.pop(0)
            if len(level["child_list"]) > 1:
                static_dim = 0
                for sub_level in level["child_list"]:
                    involved_comps = cls.get_involved_comps(sub_level)
                    for i in involved_comps:
                        comp_index = (
                            program_json["computations"][i]["absolute_order"] - 1
                        )
                        result[comp_index].append(static_dim)
                    static_dim = static_dim + 10
            else:
                if len(level["child_list"]) == 1:
                    involved_comps = cls.get_involved_comps(level)
                    for i in involved_comps:
                        comp_index = (
                            program_json["computations"][i]["absolute_order"] - 1
                        )
                        result[comp_index].append(0)
                else:
                    if len(level["computations_list"]) > 0:
                        involved_comps = cls.get_involved_comps(level)
                        static_dim = 0
                        for i in involved_comps:
                            comp_index = (
                                program_json["computations"][i]["absolute_order"] - 1
                            )
                            result[comp_index].append(static_dim)
                            static_dim = static_dim + 10
            for element in level["child_list"]:
                to_explore.append(element)
        return result

    @classmethod
    def get_involved_comps_from_iterator(cls, iterator, program_json):
        result = []
        node = program_json["iterators"][iterator]
        if len(node) == 0:
            return result
        for comp in node["computations_list"]:
            result.append(comp)
        for child in node["child_iterators"]:
            for comp in cls.get_involved_comps_from_iterator(child, program_json):
                result.append(comp)
        return result

    @classmethod
    def get_involved_comps(cls, node):
        result = []
        if len(node) == 0:
            return result
        for comp in node["computations_list"]:
            result.append(comp)
        for child in node["child_list"]:
            for comp in cls.get_involved_comps(child):
                result.append(comp)
        return result

    @classmethod
    def add_static_dims(cls, matrix, static_dims):
        size = len(matrix) * 2 + 2
        gen_matrix = np.zeros((size, size), int)
        np.fill_diagonal(gen_matrix, 1)
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                gen_matrix[2 * i + 1][2 * j + 1] = matrix[i][j]
        for i in range(len(static_dims)):
            gen_matrix[2 * i + 2][size - 1] = static_dims[i]
            gen_matrix[2 * i + 2][2 * i + 2] = 0
        return gen_matrix

    @classmethod
    def get_comp_iterators_from_tree_struct(cls, schedule_json, comp_name):
        tree = schedule_json["tree_structure"]
        level = tree
        iterators = []
        to_explore = [root for root in tree["roots"]]
        # to_explore.append(tree)
        while to_explore:
            level = to_explore.pop(0)
            if comp_name in cls.get_involved_comps(level):
                iterators.append(level["loop_name"])

            for element in level["child_list"]:
                to_explore.append(element)

        return iterators

    @classmethod
    def get_iterators_from_tree_struct(cls, schedule_json):
        tree = schedule_json["tree_structure"]
        level = tree
        iterators = []
        to_explore = []
        to_explore.append(tree)
        while to_explore:
            level = to_explore.pop(0)

            iterators.append(level["loop_name"])

            for element in level["child_list"]:
                to_explore.append(element)

        return iterators

    @classmethod
    def format_bound(cls, iterator_name, bound, iterators_list, is_lower):
        output = []
        for i in iterators_list:
            if i == iterator_name:
                if is_lower:
                    output.append(-1)
                else:
                    output.append(1)
            elif i == bound:
                if is_lower:
                    output.append(1)
                else:
                    output.append(-1)
            else:
                output.append(0)
        return output

    @classmethod
    def get_padded_initial_constrain_matrix(
        cls, program_json, schedule_json, comp_name, max_depth
    ):
        iterators_list = program_json["computations"][comp_name]["iterators"]
        result = []
        for i in iterators_list:
            for j in range(2):
                if j == 0:
                    result.append(
                        cls.format_bound(
                            i,
                            program_json["iterators"][i]["lower_bound"],
                            iterators_list,
                            True,
                        )
                    )
                else:
                    result.append(
                        cls.format_bound(
                            i,
                            program_json["iterators"][i]["upper_bound"],
                            iterators_list,
                            False,
                        )
                    )

        result = np.c_[np.ones(len(result)), result]
        result = np.r_[[np.ones(len(result[0]))], result]
        result = np.pad(
            result,
            [
                (0, (max_depth + 1) * 2 - result.shape[0]),
                (0, max_depth + 1 - result.shape[1]),
            ],
            mode="constant",
            constant_values=0,
        )
        return result

    @classmethod
    def get_padded_transformed_constrain_matrix(
        cls, program_json, schedule_json, comp_name, max_depth
    ):
        iterators_list = program_json["computations"][comp_name]["iterators"]
        transformation_matrix = cls.get_transformation_matrix(
            program_json, schedule_json, comp_name, max_depth
        )
        result = []
        for i in iterators_list:
            for j in range(2):
                if j == 0:
                    result.append(
                        cls.format_bound(
                            i,
                            program_json["iterators"][i]["lower_bound"],
                            iterators_list,
                            True,
                        )
                    )
                else:
                    result.append(
                        cls.format_bound(
                            i,
                            program_json["iterators"][i]["upper_bound"],
                            iterators_list,
                            False,
                        )
                    )
        inverse = np.linalg.inv(transformation_matrix)
        result = np.matmul(result, inverse)

        result = np.c_[np.ones(len(result)), result]
        result = np.r_[[np.ones(len(result[0]))], result]
        result = np.pad(
            result,
            [
                (0, (max_depth + 1) * 2 - result.shape[0]),
                (0, max_depth + 1 - result.shape[1]),
            ],
            mode="constant",
            constant_values=0,
        )
        return result

    @classmethod
    def get_trasnformation_matrix_from_vector(cls, transformation, matrix_size):
        matrix = np.identity(matrix_size)

        if transformation[0] == 1:
            assert transformation[1] < matrix_size and transformation[2] < matrix_size
            matrix[transformation[1], transformation[2]] = 1
            matrix[transformation[1], transformation[1]] = 0
            matrix[transformation[2], transformation[1]] = 1
            matrix[transformation[2], transformation[2]] = 0

        elif transformation[0] == 2:
            assert transformation[3] < matrix_size
            matrix[transformation[3], transformation[3]] = -1

        elif transformation[0] == 3:
            assert transformation[4] < matrix_size and transformation[5] < matrix_size
            matrix[transformation[4], transformation[4]] = transformation[6]
            matrix[transformation[4], transformation[5]] = transformation[7]

        return matrix

    @classmethod
    # transform the vectors into a series of matrices
    def get_transformation_matrix(
        cls, program_json, schedule_json, comp_name, max_depth=None
    ):
        nb_iterators = len(program_json["computations"][comp_name]["iterators"])
        final_transformation = np.identity(nb_iterators)
        for transformation in schedule_json[comp_name]["transformations_list"]:
            matrix = cls.get_trasnformation_matrix_from_vector(
                transformation, nb_iterators
            )
            final_transformation = np.matmul(matrix, final_transformation)
        return final_transformation

    @classmethod
    def get_schedule_representation(
        cls,
        schedule_object,
        max_depth=MAX_DEPTH,
    ):
        program_json = schedule_object.prog.annotations
        schedule_json = schedule_object.schedule_dict
        comps_repr_templates_list = schedule_object.repr.comps_repr_templates_list
        loops_repr_templates_list = schedule_object.repr.loops_repr_templates_list
        comps_placeholders_indices_dict = (
            schedule_object.repr.comps_placeholders_indices_dict
        )
        loops_placeholders_indices_dict = (
            schedule_object.repr.loops_placeholders_indices_dict
        )
        # Create a copy of the templates to avoid modifying the values for other schedules
        comps_repr = copy.deepcopy(comps_repr_templates_list)
        loops_repr = copy.deepcopy(loops_repr_templates_list)

        computations_dict = program_json["computations"]
        ordered_comp_list = sorted(
            list(computations_dict.keys()),
            key=lambda x: computations_dict[x]["absolute_order"],
        )

        for comp_index, comp_name in enumerate(ordered_comp_list):
            comp_dict = program_json["computations"][comp_name]
            comp_schedule_dict = schedule_json[comp_name]

            fused_levels = []

            # If fusion was applied, save which two loops were fused together
            if "fusions" in schedule_json and schedule_json["fusions"]:
                for fusion in schedule_json["fusions"]:
                    if comp_name in fusion:
                        fused_levels.append(fusion[2])

            c_code = "C" + str(comp_index)
            # Loop representation for this computation
            for iter_i, iterator_name in enumerate(comp_dict["iterators"]):
                l_code = c_code + "-L" + str(iter_i)

                # Check whether parallelization was applied and put the tag in its corresponding position in the computation representation
                parallelized = 0
                if iterator_name == comp_schedule_dict["parallelized_dim"]:
                    parallelized = 1
                p_index = comps_placeholders_indices_dict[l_code + "Parallelized"]
                comps_repr[p_index[0]][p_index[1]] = parallelized

                # Check whether tiling was applied and put the tags in their corresponding position in the computation representation
                tiled = 0
                tile_factor = 0
                if comp_schedule_dict["tiling"] and (
                    iterator_name in comp_schedule_dict["tiling"]["tiling_dims"]
                ):
                    tiled = 1
                    tile_factor_index = comp_schedule_dict["tiling"][
                        "tiling_dims"
                    ].index(iterator_name)
                    tile_factor = int(
                        comp_schedule_dict["tiling"]["tiling_factors"][
                            tile_factor_index
                        ]
                    )
                p_index = comps_placeholders_indices_dict[l_code + "Tiled"]
                comps_repr[p_index[0]][p_index[1]] = tiled
                p_index = comps_placeholders_indices_dict[l_code + "TileFactor"]
                comps_repr[p_index[0]][p_index[1]] = tile_factor

                # Check whether fusion was applied and put the tag in its corresponding position in the computation representation
                fused = 0
                if iter_i in fused_levels:
                    fused = 1
                p_index = comps_placeholders_indices_dict[l_code + "Fused"]
                comps_repr[p_index[0]][p_index[1]] = fused

                shifted = 0
                shifting_factor = 0
                if comp_schedule_dict["shiftings"]:
                    for shifting in comp_schedule_dict["shiftings"]:
                        if iterator_name.startswith(
                            shifting[0]
                        ):  # loof if the current loop is being shifted
                            shifted = 1
                            shifting_factor = shifting[1]
                            break
                p_index = comps_placeholders_indices_dict[l_code + "Shifted"]
                comps_repr[p_index[0]][p_index[1]] = shifted
                p_index = comps_placeholders_indices_dict[l_code + "ShiftFactor"]
                comps_repr[p_index[0]][p_index[1]] = shifting_factor
            # Check whether unrolling was applied and put the tags in their corresponding position in the computation representation
            unrolled = 0
            unroll_factor = 0
            if comp_schedule_dict["unrolling_factor"]:
                unrolled = 1
                unroll_factor = int(comp_schedule_dict["unrolling_factor"])

            p_index = comps_placeholders_indices_dict[c_code + "-Unrolled"]
            comps_repr[p_index[0]][p_index[1]] = unrolled

            p_index = comps_placeholders_indices_dict[c_code + "-UnrollFactor"]
            comps_repr[p_index[0]][p_index[1]] = unroll_factor

            # Check which transformations (interchange, reversal and skweing) were applied and add the padded vector representation to their corresponding position
            padded_tags = cls.get_padded_transformation_tags(
                program_json, schedule_json, comp_name, max_depth
            )

            tags_start = comps_placeholders_indices_dict[
                c_code + "-TransformationTagsStart"
            ]

            tags_end = comps_placeholders_indices_dict[
                c_code + "-TransformationTagsEnd"
            ]

            nb_tags_elements = tags_end[1] - tags_start[1] + 1

            assert len(padded_tags) == nb_tags_elements

            comps_repr[tags_start[0]][tags_start[1] : tags_end[1] + 1] = padded_tags

            ogc_start = comps_placeholders_indices_dict[
                c_code + "-OgConstraintMatrixStart"
            ]

            ogc_end = comps_placeholders_indices_dict[c_code + "-OgConstraintMatrixEnd"]

            nb_mat_elements = ogc_end[1] - ogc_start[1] + 1

            max_depth_it = int(np.sqrt(nb_mat_elements / 2)) - 1

            comps_repr[ogc_start[0]][ogc_start[1] : ogc_end[1] + 1] = (
                cls.get_padded_initial_constrain_matrix(
                    program_json, schedule_json, comp_name, max_depth_it
                )
                .flatten()
                .tolist()
            )

            ogv_start = comps_placeholders_indices_dict[
                c_code + "-OgConstraintVectorStart"
            ]

            ogv_end = comps_placeholders_indices_dict[c_code + "-OgConstraintVectorEnd"]

            nb_mat_elements = ogv_end[1] - ogv_start[1] + 1

            comps_repr[ogv_start[0]][
                ogv_start[1] : ogv_end[1] + 1
            ] = cls.get_padded_second_side_of_the_constraint_equations_original(
                program_json, schedule_json, comp_name, max_depth
            )

            c_start = comps_placeholders_indices_dict[c_code + "-ConstraintMatrixStart"]

            c_end = comps_placeholders_indices_dict[c_code + "-ConstraintMatrixEnd"]

            nb_mat_elements = c_end[1] - c_start[1] + 1

            max_depth_it = int(np.sqrt(nb_mat_elements / 2)) - 1

            comps_repr[c_start[0]][c_start[1] : c_end[1] + 1] = (
                cls.get_padded_transformed_constrain_matrix(
                    program_json, schedule_json, comp_name, max_depth_it
                )
                .flatten()
                .tolist()
            )

        # Fill the loop representation
        # Initialization
        loop_schedules_dict = dict()
        for loop_name in program_json["iterators"]:
            loop_schedules_dict[loop_name] = dict()
            loop_schedules_dict[loop_name]["tiled"] = 0
            loop_schedules_dict[loop_name]["tile_factor"] = 0
            loop_schedules_dict[loop_name]["unrolled"] = 0
            loop_schedules_dict[loop_name]["unroll_factor"] = 0
            loop_schedules_dict[loop_name]["shifted"] = 0
            loop_schedules_dict[loop_name]["shift_factor"] = 0
            loop_schedules_dict[loop_name]["parallelized"] = 0
            loop_schedules_dict[loop_name]["fused"] = 0

        for comp_index, comp_name in enumerate(ordered_comp_list):
            comp_schedule_dict = schedule_json[comp_name]

            # Check whether tiling was applied
            if comp_schedule_dict["tiling"]:
                for tiled_loop_index, tiled_loop in enumerate(
                    comp_schedule_dict["tiling"]["tiling_dims"]
                ):
                    loop_schedules_dict[tiled_loop]["tiled"] = 1
                    assert loop_schedules_dict[tiled_loop][
                        "tile_factor"
                    ] == 0 or loop_schedules_dict[tiled_loop]["tile_factor"] == int(
                        comp_schedule_dict["tiling"]["tiling_factors"][tiled_loop_index]
                    )
                    loop_schedules_dict[tiled_loop]["tile_factor"] = int(
                        comp_schedule_dict["tiling"]["tiling_factors"][tiled_loop_index]
                    )

            # Check whether unrolling was applied
            if comp_schedule_dict["unrolling_factor"]:
                comp_innermost_loop = cls.get_comp_iterators_from_tree_struct(
                    schedule_json, comp_name
                )[-1]
                #             comp_innermost_loop = computations_dict[comp_name]["iterators"][-1]
                loop_schedules_dict[comp_innermost_loop]["unrolled"] = 1

                assert loop_schedules_dict[comp_innermost_loop][
                    "unroll_factor"
                ] == 0 or loop_schedules_dict[comp_innermost_loop][
                    "unroll_factor"
                ] == int(
                    comp_schedule_dict["unrolling_factor"]
                )

                loop_schedules_dict[comp_innermost_loop]["unroll_factor"] = int(
                    comp_schedule_dict["unrolling_factor"]
                )

            # Check whether parallelization was applied
            if comp_schedule_dict["parallelized_dim"]:
                loop_schedules_dict[comp_schedule_dict["parallelized_dim"]][
                    "parallelized"
                ] = 1

            if comp_schedule_dict["shiftings"]:
                for shifting in comp_schedule_dict["shiftings"]:
                    loop_schedules_dict[shifting[0]]["shifted"] = 1
                    loop_schedules_dict[shifting[0]]["shift_factor"] = shifting[1]

        # Check whether fusion was applied
        if "fusions" in schedule_json and schedule_json["fusions"]:
            for fusion in schedule_json["fusions"]:
                fused_loop1 = computations_dict[fusion[0]]["iterators"][fusion[2]]
                fused_loop2 = computations_dict[fusion[1]]["iterators"][fusion[2]]
                loop_schedules_dict[fused_loop1]["fused"] = 1
                loop_schedules_dict[fused_loop2]["fused"] = 1

        program_iterators = cls.get_comp_iterators_from_tree_struct(
            schedule_json, comp_name
        )
        # Get the index of each feature in the loop representation and replace it with the the information obtained from the schedule
        for loop_name in program_json["iterators"]:
            l_code = "L" + loop_name

            p_index = loops_placeholders_indices_dict[l_code + "Parallelized"]
            loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name][
                "parallelized"
            ]

            p_index = loops_placeholders_indices_dict[l_code + "Tiled"]
            loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name]["tiled"]
            p_index = loops_placeholders_indices_dict[l_code + "TileFactor"]
            loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name][
                "tile_factor"
            ]

            p_index = loops_placeholders_indices_dict[l_code + "Unrolled"]
            loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name][
                "unrolled"
            ]
            p_index = loops_placeholders_indices_dict[l_code + "UnrollFactor"]
            loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name][
                "unroll_factor"
            ]

            p_index = loops_placeholders_indices_dict[l_code + "Shifted"]
            loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name][
                "shifted"
            ]
            p_index = loops_placeholders_indices_dict[l_code + "ShiftFactor"]
            loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name][
                "shift_factor"
            ]

            p_index = loops_placeholders_indices_dict[l_code + "Fused"]
            loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name]["fused"]

        if len(program_json["iterators"]) > len(program_iterators):
            removed_iterators = list(
                set(program_json["iterators"]) - set(program_iterators)
            )
            for loop_name in removed_iterators:
                l_code = "L" + loop_name

                p_index = loops_placeholders_indices_dict[l_code + "Parallelized"]
                loops_repr[p_index[0]][p_index[1]] = 0

                p_index = loops_placeholders_indices_dict[l_code + "Tiled"]
                loops_repr[p_index[0]][p_index[1]] = 0
                p_index = loops_placeholders_indices_dict[l_code + "TileFactor"]
                loops_repr[p_index[0]][p_index[1]] = 0

                p_index = loops_placeholders_indices_dict[l_code + "Unrolled"]
                loops_repr[p_index[0]][p_index[1]] = 0
                p_index = loops_placeholders_indices_dict[l_code + "UnrollFactor"]
                loops_repr[p_index[0]][p_index[1]] = 0

                p_index = loops_placeholders_indices_dict[l_code + "Shifted"]
                loops_repr[p_index[0]][p_index[1]] = 0
                p_index = loops_placeholders_indices_dict[l_code + "ShiftFactor"]
                loops_repr[p_index[0]][p_index[1]] = 0

                p_index = loops_placeholders_indices_dict[l_code + "Fused"]
                loops_repr[p_index[0]][p_index[1]] = 0

        loops_tensor = torch.unsqueeze(torch.FloatTensor(loops_repr), 0)
        computations_tensor = torch.unsqueeze(torch.FloatTensor(comps_repr), 0)

        return computations_tensor, loops_tensor

    @classmethod
    # check whether the string contains an integer and return true if so
    def is_int(cls, s):
        # TODO : in our current case we have always integers so it always true,But this should be removed
        return True
        if s[0] in ("-", "+"):
            return s[1:].isdigit()
        return s.isdigit()

    # returns a vector that represents the right hand sise of teh constraint matrix inequalities
    # returns b where: Ax <= b and A being the constarint matrix
    @classmethod
    def get_padded_second_side_of_the_constraint_equations_original(
        cls, program_json, schedule_json, comp_name, max_depth
    ):
        iterators_list = program_json["computations"][comp_name]["iterators"]
        result = []
        for it in iterators_list:
            if cls.is_int(program_json["iterators"][it]["lower_bound"]):
                result.append(int(program_json["iterators"][it]["lower_bound"]))
            else:
                result.append(0)
            if cls.is_int(program_json["iterators"][it]["upper_bound"]):
                result.append(int(program_json["iterators"][it]["upper_bound"]))
            else:
                result.append(0)
        result = result + [0] * (max_depth * 2 - len(result))
        return result

    @classmethod
    def get_padded_transformation_tags(
        cls, program_json, schedule_json, comp_name, max_depth=None
    ):
        # Extract information about the computation and the transformations that were applied from the json input
        comp_dict = program_json["computations"][comp_name]
        comp_schedule_dict = schedule_json[comp_name]
        nb_iterators = len(comp_dict["iterators"])
        loop_nest = comp_dict["iterators"][:]

        # Create an identity vector that represents that no transformation was applied
        identity = np.zeros((nb_iterators, nb_iterators), int)
        np.fill_diagonal(identity, 1)
        identity_tags = np.zeros((1, MAX_TAGS), dtype=np.int32)

        tag_factors = []
        for transformation in comp_schedule_dict["transformations_list"]:
            tag_factors.append(transformation)

        # Add padding to the sequence of vectors in case the number of transformations is less than MAX_NUM_TRANSFORMATIONS+1
        tags_list = [item for sublist in tag_factors for item in sublist]
        tags_list += [0] * (MAX_NUM_TRANSFORMATIONS * MAX_TAGS - len(tags_list))

        return tags_list

    @classmethod
    def seperate_vector(
        cls,
        X: torch.Tensor,
        num_transformations: int = 4,
        pad: bool = True,
        pad_amount: int = 5,
    ) -> torch.Tensor:
        batch_size, _ = X.shape
        first_part = X[:, :33]
        second_part = X[:, 33 : 33 + MAX_TAGS * num_transformations]
        third_part = X[:, 33 + MAX_TAGS * num_transformations :]
        vectors = []
        for i in range(num_transformations):
            vector = second_part[:, MAX_TAGS * i : MAX_TAGS * (i + 1)].reshape(
                batch_size, 1, -1
            )
            vectors.append(vector)

        if pad:
            for i in range(pad_amount):
                vector = torch.zeros_like(vector)
                vectors.append(vector)
        return (first_part, torch.cat(vectors[0:], dim=1), third_part)

    @classmethod
    def get_tree_representation(
        cls, comps_tensor, loops_tensor, expr_tensor, schedule_object
    ):
        """
        This function returns the necessary dict and tensors to feed the cost model to get the speedup
        """
        x = comps_tensor
        batch_size, num_comps, __dict__ = x.shape
        x = x.view(batch_size * num_comps, -1)
        (first_part, vectors, third_part) = ConvertService.seperate_vector(
            x, num_transformations=4, pad=False
        )
        first_part = first_part.view(batch_size, num_comps, -1)
        third_part = third_part.view(batch_size, num_comps, -1)
        return (
            schedule_object.repr.prog_tree,
            first_part,
            vectors,
            third_part,
            loops_tensor,
            expr_tensor,
        )

    @classmethod
    def get_encoded_rl_representation(cls, comps_tensor, loops_tensor, schedule_object):
        """
        Becaue currently in rllib they don't support gymnasium latest data strucutres (spaces.Text snd spaces.Sequence),
        it is necessary to encode the fields into a fixed format that contains just numbers
        """
        encoded_tree = np.array(
            cls.encode_dict(schedule_object.repr.prog_tree), dtype=np.float32
        )
        encoded_comps = cls.pad_3D_vector(
            comps_tensor.detach().numpy().astype("float32")
        )
        encoded_loops = cls.pad_3D_vector(
            loops_tensor.detach().numpy().astype("float32"), padding=32
        )
        encoded_comps_expr = cls.pad_4D_vector(
            schedule_object.repr.comps_expr_tensor.detach().numpy().astype("float32")
        )
        encoded_expr_loops = cls.pad_1D_vector(
            schedule_object.repr.comps_expr_lengths.detach().numpy().astype("float32")
        )
        return (
            encoded_tree,
            encoded_comps,
            encoded_loops,
            encoded_comps_expr,
            encoded_expr_loops,
        )

    @classmethod
    def get_decoded_rl_repr(
        cls,
        encoded_tree,
        encoded_comps,
        encoded_loops,
        encoded_comps_expr,
        encoded_expr_loops,
    ):
        """
        After passing the coded representation in the gym and rllib envireonment , we will use this function to decode the numerical representation
        into the objects that's going to feed the policy model
        """
        decoded_tree = cls.decode_dict(encoded_tree.astype("int32").tolist())
        decoded_comps = torch.from_numpy(cls.unpad_3D_vector(encoded_comps))
        decoded_loops = torch.from_numpy(cls.unpad_3D_vector(encoded_loops))
        decoded_comps_expr = torch.from_numpy(cls.unpad_4D_vector(encoded_comps_expr))
        decoded_expr_loops = torch.from_numpy(
            cls.unpad_1D_vector(encoded_expr_loops).astype("int32")
        )
        x = decoded_comps
        batch_size, num_comps, __dict__ = x.shape
        x = x.view(batch_size * num_comps, -1)
        (first_part, vectors, third_part) = ConvertService.seperate_vector(
            x, num_transformations=4, pad=False
        )
        first_part = first_part.view(batch_size, num_comps, -1)
        third_part = third_part.view(batch_size, num_comps, -1)

        return (
            decoded_tree,
            first_part,
            vectors,
            third_part,
            decoded_loops,
            decoded_comps_expr,
            decoded_expr_loops,
        )

    @classmethod
    def encode_dict(cls, json_dict):
        """
        This class converts a dict into a fixed-size array of ascii encoded chars
        TODO : This function should be removed when rllib supports the spaces
        """
        # Convert the JSON dictionary to a string
        json_string = json.dumps(json_dict)

        # Convert the string to an array of ASCII numbers
        ascii_array = [ord(char) for char in json_string]

        # Pad the array to a fixed size
        fixed_size = 2000
        padded_array = ascii_array + [0] * (fixed_size - len(ascii_array))

        # Return the padded array
        return padded_array

    @classmethod
    def decode_dict(cls, padded_array):
        """
        This class does the inverse of the above class
        TODO : This function should be removed when rllib supports the spaces
        """
        # Remove the padding
        trimmed_array = padded_array[: padded_array.index(0)]

        # Convert the array of ASCII numbers back into a string
        ascii_string = "".join([chr(num) for num in trimmed_array])

        # Convert the string back into a JSON dictionary
        json_dict = json.loads(ascii_string)

        # Return the resulting JSON dictionary
        return json_dict

    @classmethod
    def pad_1D_vector(cls, vector, padding=16):
        x = len(vector)
        padded_vector = np.pad(vector, (0, padding - x), constant_values=np.inf)
        return padded_vector

    @classmethod
    def unpad_1D_vector(cls, padded_array):
        first_padded_index = np.where(padded_array == np.inf)[0][0]

        # unpad the array by slicing up to the first padded element
        return padded_array[:first_padded_index]

    @classmethod
    def pad_3D_vector(cls, vector, padding=10):
        _, x, _ = vector.shape
        padded_vector = np.pad(
            vector, ((0, 0), (0, padding - x), (0, 0)), constant_values=np.inf
        )
        return padded_vector

    @classmethod
    def unpad_3D_vector(cls, padded_array):
        # Find the index of the first all-zero vector along the 2nd axis
        all_zeros = np.all(padded_array == np.inf, axis=(0, 2))
        first_zero_index = np.argmax(all_zeros)

        # Remove the all-zero vectors from the 2nd axis
        return np.delete(padded_array, np.s_[first_zero_index:], axis=1)

    @classmethod
    def pad_4D_vector(cls, vector):
        _, x, y, _ = vector.shape
        padded_vector = np.pad(
            vector, ((0, 0), (0, 5 - x), (0, 64 - y), (0, 0)), constant_values=np.inf
        )
        return padded_vector

    @classmethod
    def unpad_4D_vector(cls, padded_array):
        # Find the index of the first all-zero vector along the 2nd axis
        all_zeros = np.all(padded_array == np.inf, axis=(0, 3))
        first_zero_index = np.argmax(all_zeros)

        # Remove the all-zero vectors from the 2nd axis
        trimmed_y = np.delete(padded_array, np.s_[first_zero_index:], axis=2)
        gt = []
        for z in trimmed_y[0]:
            if z[0][0] != np.inf:
                gt.append(z.astype("float32").tolist())
        return np.array([gt])

    # TODO : add fusion schedule
    @classmethod
    def build_sched_string(cls, schedule_list) -> str:
        # Prepare a dictionary of computations name to fill it with each action applied on every comp
        comps = {}
        # Map the schedules applied one by one
        for schedule in schedule_list:
            # schedule has comps_schedule which includes the comps that was invloved in the optimisation
            for key in schedule.comps_schedule.keys():
                # Add the data from that schedule to the global comps dictionnary
                if not key in comps or not comps[key]:
                    comps[key] = ""
                comps[key] += schedule.comps_schedule[key]

        # Prepare the string and form it from the comps dictionary
        schedule_string = ""

        # Check if the first optimization applied is fusion
        if schedule_list and type(schedule_list[0].action) == Fusion:
            schedule_string += schedule_list[0].fusion_str + ":"

        for key in comps.keys():
            schedule_string += "{" + key + "}:" + comps[key]
        return schedule_string

    @classmethod
    def build_loop_nests(cls, parent, iterators):
        tree = {}
        tree["loop_name"] = parent
        tree["computations_list"] = iterators[parent]["computations_list"]
        child_trees = []
        if iterators[parent]["child_iterators"]:
            for iterator in iterators[parent]["child_iterators"]:
                child_trees.append(cls.build_loop_nests(iterator, iterators))

        tree["child_list"] = child_trees
        return tree

    @classmethod
    def build_tree_structure(cls, iters):
        iterators = copy.deepcopy(iters)
        roots = []
        for iterator in iterators:
            if not iterators[iterator]["parent_iterator"]:
                roots.append(copy.copy(iterator))

        return [cls.build_loop_nests(it, iterators) for it in roots]
