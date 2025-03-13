def test_subtree(tree_results : dict, alpha : float, threshold : float, direction : str) :
    assert tree_results["confidence_interval"] is not None
    lower_bound, upper_bound = tree_results["confidence_interval"][str(alpha)]
    if direction == "higher" :
        return lower_bound > threshold
    elif direction == "lower" :
        return upper_bound < threshold
    else :
        raise NotImplementedError("direction = {}".format(direction))


def extract_subtrees(tree_results : dict, alpha : float, threshold : float, direction : str, extracted : bool = False) :
    if extracted :
        tree_results["extracted"] = False
    else :
        if tree_results["confidence_interval"] is None :
            tree_results["extracted"] = False
        else :
            if test_subtree(tree_results, alpha, threshold, direction) :
                extracted = True
                for subtree_results in tree_results["subtrees"] if isinstance(tree_results["subtrees"], list) else tree_results["subtrees"].values() :
                    if subtree_results["size"] >= 20 and (not test_subtree(subtree_results, alpha, threshold, direction)) : # We want to find a more specific capability by going down
                        extracted = False
                        break
                tree_results["extracted"] = extracted
            else :
                tree_results["extracted"] = False
    if not isinstance(tree_results["subtrees"], int) :
        for subtree_results in tree_results["subtrees"] if isinstance(tree_results["subtrees"], list) else tree_results["subtrees"].values() :
            extract_subtrees(subtree_results, alpha, threshold, direction, extracted)
