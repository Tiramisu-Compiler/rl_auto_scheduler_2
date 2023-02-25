class Representation:
    def __init__(self,prog_tree,
    comps_repr_templates_list,
    loops_repr_templates_list,
    comps_placeholders_indices_dict,
    loops_placeholders_indices_dict,
    comps_expr_tensor,
    comps_expr_lengths):
        self.prog_tree= prog_tree
        self.comps_repr_templates_list=comps_repr_templates_list
        self.loops_repr_templates_list=loops_repr_templates_list
        self.comps_placeholders_indices_dict=comps_placeholders_indices_dict
        self.loops_placeholders_indices_dict=loops_placeholders_indices_dict
        self.comps_expr_tensor=comps_expr_tensor
        self.comps_expr_lengths=comps_expr_lengths
        self.action_mask = None