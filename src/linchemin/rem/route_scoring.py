from abc import ABC, abstractmethod

from linchemin.rem.route_descriptors import descriptor_calculator

"""
Module containing classes and functions to score SynRoutes.

    AbstractClasses:
        RouteScore
        
    Classes:
        ScoreFactory
        
        BranchScore(RouteScore)
        
    Functions:
        route_scorer(syngraph: SynGraph, score: str)
"""


class RouteScore(ABC):
    """Definition of the abstract class for RouteScore."""

    @abstractmethod
    def compute_score(self, syngraph):
        pass


class BranchScore(RouteScore):
    """Subclass of RouteScore representing the branchedness based score of a SynGraph."""

    def compute_score(self, syngraph):
        """Takes a SynGraph and returns its branchedness score. The score is the ratio between the branchedness
        of the route and the ideal branchedness for a route with the same number of steps.
        """
        actual_branchedness = descriptor_calculator(syngraph, "branchedness")
        n_steps = descriptor_calculator(syngraph, "nr_steps")
        ideal_branchedness = n_steps - 1.0
        if ideal_branchedness == 0:
            return 0
        return actual_branchedness / ideal_branchedness


class ScoreFactory:
    """Definition of RouteScore factory"""

    scores = {
        "branch_score": {
            "value": BranchScore,
            "info": "Assigns a score based on the route branchedness compared to the ideal value",
        },
    }

    def select_score(self, syngraph, score: str):
        """Takes a SynGraph and a string indicating a score and returns the value of the score [0, 1]"""
        if score not in self.scores:
            raise KeyError(f"Invalid score. Available scores are: {self.scores.keys()}")

        calculator = self.scores[score]["value"]
        return calculator().compute_score(syngraph)


def route_scorer(syngraph, score: str):
    """Gives access to ScoreFactory.

    Parameters:
        syngraph: a Syngraph/MonopartiteSynGraph instance
        score: a string indicating the score to be computed

    Returns:
        a float between 0 and 1 (0 being 'bad' and 1 being 'good')
    """
    score_selector = ScoreFactory()
    return score_selector.select_score(syngraph, score)
