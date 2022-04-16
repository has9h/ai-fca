    def search(self):
        """overridden function.
        returns (next) path from the problem's start node
        to a goal node. 
        Returns None if no path exists.
        """
        while not self.empty_frontier():
            path = self.frontier.pop()
            self.display(2, "Expanding:",path,"(cost:",path.cost,")")
            self.num_expanded += 1
            self.add_to_explored(path)
            if self.problem.is_goal(path.end()):    # solution found
                self.display(1, self.num_expanded, "paths have been expanded and",
                            len(self.frontier), "paths remain in the frontier")
                self.solution = path   # store the solution found
                return path
            else:
                neighs = self.problem.neighbors(path.end())
                self.display(3,"Neighbors are", neighs)
                for arc in reversed(list(neighs)):
                    if arc.to_node not in self.explored:
                        self.add_to_frontier(Path(path,arc))
                self.display(3,"Frontier:",self.frontier)
        self.display(1,"No (more) solutions. Total of",
                        self.num_expanded,"paths expanded.")

    
def start(self):
    """return the node at the start of the path"""
    if self.arc is None:
        return self.initial
    else:
        return self.arc.from_node

undirected_problem = Search_problem_from_explicit_graph(
    {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'},
    [Arc('a', 'b', 4), Arc('b', 'a', 4),
    Arc('a', 'c', 8), Arc('c', 'a', 8),
    Arc('b', 'c', 11), Arc('c', 'b', 11),
    Arc('b', 'e', 8), Arc('e', 'b', 8),
    Arc('c', 'd', 7), Arc('d', 'c', 7),
    Arc('c', 'f', 1), Arc('f', 'c', 1),
    Arc('d', 'e', 2), Arc('e', 'd', 2),
    Arc('d', 'f', 6), Arc('f', 'd', 6),
    Arc('e', 'h', 4), Arc('h', 'e', 4),
    Arc('e', 'g', 7), Arc('g', 'e', 7),
    Arc('f', 'h', 2), Arc('h', 'f', 2),
    Arc('h', 'g', 14), Arc('g', 'h', 14),
    Arc('h', 'i', 10), Arc('i', 'h', 10),
    Arc('g', 'i', 9), Arc('i', 'g', 9)
    ],
    start='a',
    goals={'d'},
    hmap={'a': 24,
          'b': 20,
          'c': 18,
          'd': 17,
          'e': 14,
          'f': 15,
          'g': 10,
          'h': 10,
          'i': 0,
          }
)