from copy import deepcopy


def get_branch_comp(node, comp_name):
    """Returns the branch of the node that contains the comp_name."""
    branch = []

    # If the comp_name is in the node
    if comp_name in node["computations_list"]:
        branch.append(node)
        return branch

    # If the node is a leaf node
    if len(node["child_list"]) == 0:
        return branch

    # If the node is not a leaf node
    for child in node["child_list"]:
        branch = get_branch_comp(child, comp_name)
        if len(branch) != 0:
            branch.append(node)
            return branch
    return branch


def transform_tree_for_fusion(original_tree, fusions):
    """Transforms the tree structure to include the fusions."""
    new_tree = deepcopy(original_tree)

    if fusions is None:
        return original_tree

    # Iterate over the fusions
    for fusion in fusions:
        # Get the level of the fusion
        level = fusion[2]

        branch_comp_0 = get_branch_comp(new_tree["roots"][0], fusion[0])[::-1]
        branch_comp_1 = get_branch_comp(new_tree["roots"][0], fusion[1])[::-1]

        if len(branch_comp_0) == 0 or len(branch_comp_1) == 0:
            continue

        if len(branch_comp_0) >= len(branch_comp_1):
            branch_comp_0[level]["computations_list"].append(fusion[1])
            branch_comp_1[-1]["computations_list"].remove(fusion[1])

            # Remove the empty nodes from new_tree
            for i in range(level, 0, -1):
                if (
                    len(branch_comp_1[i]["computations_list"]) == 0
                    and len(branch_comp_1[i]["child_list"]) == 0
                ):
                    branch_comp_1[i - 1]["child_list"].remove(branch_comp_1[i])
        else:
            branch_comp_1[level]["computations_list"].append(fusion[0])
            branch_comp_0[-1]["computations_list"].remove(fusion[0])

            # Remove the empty nodes from new_tree
            for i in range(level, 0, -1):
                if (
                    len(branch_comp_0[i]["computations_list"]) == 0
                    and len(branch_comp_0[i]["child_list"]) == 0
                ):
                    branch_comp_0[i - 1]["child_list"].remove(branch_comp_0[i])

    return new_tree
