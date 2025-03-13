def get_capability2performance(capabilities : list, assignments : list[dict], results, results_type) :
    assert len(assignments) == len(results)
    capability2performance = {capability : [] for capability in capabilities}
    for assignment, result in zip(assignments, results) :
        for capability in assignment["assignment"] :
            capability = capabilities[int(capability) - 1]
            if results_type == "accuracy" :
                assert isinstance(result, int)
                capability2performance[capability].append(result)
            elif results_type == "win-rate" :
                assert isinstance(result, list) and len(result) == 2
                capability2performance[capability].append((int(result[0] == 1) + int(result[1] == 1)) / 2.0)
            else :
                raise NotImplementedError("results_type = {}".format(results_type))

    for capability, performance in capability2performance.items() :
        capability2performance[capability] = sum(performance) / len(performance)
    return capability2performance


def get_capability2performance_split(capabilities : list, assignments : list[dict], results, results_type, split) :
    return get_capability2performance(capabilities, [assignments[index] for index in split], [results[index] for index in split], results_type)