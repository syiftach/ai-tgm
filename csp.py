from table_guest_management import *


class CSP:
    """
    this class represents a CSP agent that solve the TGM assignment problem
    """
    def __init__(self, variables, domains, agent: TableGuestManager):
        # list of variables that need to be assigned a value from their respective domain
        self.variables = variables.copy()
        # dict of domains, variable maps to its respective domain
        dom_copy = domains.copy()
        # we shuffle the tables domain, because we don't want the order to affect the csp assignment. specifically,
        # give it advantage over other agents.
        np.random.seed(0)
        np.random.shuffle(dom_copy)
        self.domains = {var: dom_copy.copy() for var in self.variables}
        # mapping from every scope to its corresponding list of constraints
        self.constraints = {var: [] for var in self.variables}
        # problem outside agent
        self.agent = agent
        assert len(self.variables) == len(self.domains)

    def add_constraint(self, scope, relation):
        """
        add constraint to the set of constraints
        :param scope: Reservation, the instance that has some constraint over
        :param relation: callable, a function that given an value (assigment of the scope), return boolean, indicating
            if the constraint is satisfied
        :return:
        """
        self.constraints[scope].append(Constraint(scope, relation, self.agent))

    def init_constraints(self, relations):
        """
        add constraints to all variables and make their respective domains consistent
        :param relations: list[callable], relations to be satisfied by the constraint
        :return:
        """
        for scope in self.variables:
            for relation in relations:
                self.add_constraint(scope, relation)
        # make all nodes consistent
        self._make_nodes_consistent()

    def is_consistent(self, variable, value):
        """
        check if the given value (list of tables) is a consistent assignment for the given variable (Reservation).
        specifically: all the constraints over the reservation are satisfied

        :param variable: Reservation, reservation for which we check the consistency.
        :param value: list[Table], assignment for the reservation
        :return: boolean, True if the assignment is consistent, False other wise
        """
        assert variable in self.constraints.keys()
        for constraint in self.constraints[variable]:
            if not constraint.is_satisfied(value):
                return False
        return True

    def _make_nodes_consistent(self):
        """
        makes all the scopes (Reservations) of the CSP problem to be consistent: all values in each reservation's
        respective domain is a consistent assignment
        :return:
        """
        for variable in self.variables:
            self.domains[variable] = list(
                filter(lambda table: self.agent.get_table_max_size(table) >= variable.people, self.domains[variable]))
            self.domains[variable] = list(filter(lambda table: table.area == variable.area, self.domains[variable]))

    def _all_nodes_have_assignment(self):
        """
        check if all scopes (reservation) of the CSP problem have at least one possible consistent assignment.
        otherwise, the CSP problem is not solve-able
        :return: True if the every scope has at least one consistent assignment, False otherwise
        """
        for variable in self.variables:
            if len(self.domains[variable]) == 0:
                return False
        return True

    def _get_minimum_remaining_value_variables(self, unassigned):
        """
        MRV heuristic that search for the scopes with the minimal number of possible assignments
        :param unassigned: list[Reservation], un-assigned reservations
        :return: ndarray[Reservation], list of reservation with the minimal number of possible assignment
        """
        values_amount = [len(self.domains[variable]) for variable in unassigned]
        all_min_idx = np.argwhere(values_amount == np.min(values_amount)).flatten()
        return np.asarray(unassigned)[all_min_idx]

    def _select_unassigned_variable(self, unassigned):
        """
        this method uses the heuristic in order to compute the next variable (Reservation) to be assigned)
        :param unassigned: list[Reservation], un-assigned reservations
        :return: Reservation, next reservation to be assigned
        """
        if len(unassigned) == 0:
            return None
        min_values_vars = self._get_minimum_remaining_value_variables(unassigned)
        if len(min_values_vars) > 1:
            return get_sorted_by_tags(min_values_vars)[0]
        else:
            return min_values_vars[0]

    def _is_assignment_complete(self, assignment):
        """
        :param assignment: dict[Reservation,list[Table]], assignment to evaluate (solution for the CSP problem)
        :return: True if the given assignment is a solution for the problem, False otherwise
        """
        return len(assignment) == len(self.variables)

    def _search_helper(self, assignment):
        """
        implements the backtracking CSP algorithm
        :param assignment: dict[Reservation,list[Table]]
        :return: dict[Reservation,list[Table]], if found solution for the problem, None otherwise
        """
        # if solution is found return the solution
        if self._is_assignment_complete(assignment):
            return assignment
        else:
            unassigned = [var for var in self.variables if var not in assignment.keys()]
            next_var = self._select_unassigned_variable(unassigned)
            for value in self.domains[next_var]:
                if self.is_consistent(next_var, value):
                    local_assignment = assignment.copy()
                    tables_group = self.agent.update_scheduler(value, next_var)
                    # assert len(tables_group) >= 1
                    local_assignment.update({next_var: tables_group})
                    result = self.backtracking_search(local_assignment)
                    # if did not fail
                    if result is not None:
                        return result
                    else:
                        # revert table assignment: set the table as free in the scheduler
                        self.agent.revert_scheduler_assignment(next_var)
            # the search failed, no consistent assignment was found. thus, there is no solution.
            return None

    def backtracking_search(self, assignment: dict):
        """
        check if the problem is solve-able before starting the backtracking search
        :param assignment: dict[Reservation,list[Table]], dictionary representing the solution for the problem
        :return:
            * solution: dict[Reservation,list[Table]], if found solution for the problem
            * None, otherwise
        """
        if self._all_nodes_have_assignment():
            return self._search_helper(assignment)
        return None


class Constraint:
    """
    this class represents a constraint over some scope
    """
    def __init__(self, scope, relation: callable, agent: TableGuestManager):
        self.scope = scope
        self.relation = relation
        self.agent = agent

    def __str__(self):
        return f'{self.relation.__name__}'

    def is_satisfied(self, value):
        """
        checks if the given value for the constraint's scope satisfies the constraint
        :param value: Table, assignment value for a Reservation
        :return: boolean, True if value satisfies, False otherwise
        """
        return self.relation(self.scope, value, self.agent)


# ============================== CONSTRAINTS_RELATIONS ============================== #


def size_constraint(scope, value, agent):
    """
    constraint relation: size constraint over a Reservation
    :param scope: Reservation, reservation to check the constraint for
    :param value: Table, value assignment for the given reservation
    :param agent: TableGuestManager, manager of the given reservation and table
    :return: boolean, True if constraint is satisfied: if the given table is large enough, False otherwise
    """
    return agent.get_table_max_size(value) >= scope.people


def availability_constraint(scope, value, agent):
    """
    constraint relation: availability constraint over a Reservation
    :param scope: Reservation, reservation to check the constraint for
    :param value: Table, value assignment for the given reservation
    :param agent: TableGuestManager, manager of the given reservation and table
    :return: boolean, True if constraint is satisfied: if the given table is available, False otherwise
    """
    return agent.is_table_free(value, scope.start, scope.end)


def area_constraint(scope, value, agent):
    """
    constraint relation: area constraint over a Reservation
    :param scope: Reservation, reservation to check the constraint for
    :param value: Table, value assignment for the given reservation
    :param agent: TableGuestManager, manager of the given reservation and table
    :return: boolean, True if constraint is satisfied: the table is the desired area, False otherwise
    """
    return value.area == scope.area


def tags_constraint(scope, value, agent):
    """
    constraint relation: tags constraint over a Reservation
    :param scope: Reservation, reservation to check the constraint for
    :param value: Table, value assignment for the given reservation
    :param agent: TableGuestManager, manager of the given reservation and table
    :return: boolean, True if constraint is satisfied: if all hard tags are satisfied, False otherwise
    """
    for tag in scope.tags:
        if tag == Tags.WHEELCHAIR and value.floor > 1:
            return False
        elif tag == Tags.BABYCHAIR and value.table_type != TableType.NORMAL:
            return False
    return True


if __name__ == '__main__':
    pass
