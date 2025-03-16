import itertools
class UniqueSubsets:
    def get_subsets(self, nums):
        return [list(subset) for i in range(len(nums)+1) for subset in itertools.combinations(nums, i)]
