from util import *
from csp import *
import itertools
import argparse


class TGMLocalSearch:
    """
    this class represents a local search optimizer for the reservation assignment problem
    """

    def __init__(self, csp_agent: CSP, tgm_agent: TableGuestManager):
        # csp agent that returns the solution we optimize
        self.csp = csp_agent
        # TableGuestManager that manages the reservations and tables of the problem
        self.tgm = tgm_agent
        # the most late date and time of some reservation (used for normalizing the the get_value method)
        self.max_date = np.max([res.created.timestamp() for res in self.csp.variables])
        # the most early date and time of some reservation (used for normalizing the the get_value method)
        self.min_date = np.min([res.created.timestamp() for res in self.csp.variables])
        # maximal number of schedule holes that can occur in the TGM's schedule
        self.max_time_holes = self.tgm.get_tables_amount() * 8
        # max table rank over the TGM's tables
        max_rank = np.max([table.rank for table in self.tgm.tables])
        # maximal date_rank score that can be achieved (used for normalizing the the get_value method)
        self.max_date_rank_score = max_rank * self.tgm.get_reservation_amount()
        # total number of soft tags amount over the set of reservations
        self.soft_tags_amount = tgm_agent.n_soft_tags

        assert set(csp_agent.variables) == set(tgm_agent.reservations)
        # assert self.max_date > 0
        for variable in csp_agent.variables:
            assert set(csp_agent.domains[variable]).issubset(tgm_agent.tables)

    def get_successors(self, state):
        """
        return the all the successive states of the given state, achieved by:
            * MOVE action: moving a table to some other available table in the restaurant
            * SWITCH action: switch between assignments of two reservations

        :param state: dict[Reservation,list[Table]], representing the current assignment
        :return: list[dict[Reservation,list[Table]]], all possible successive states
        """
        # generate successors achieved by MOVE action
        move_successors = self._possible_move_actions(state)
        # generate successors achieved by SWITCH action
        switch_successors = self._possible_switch_actions(state)
        all_moves = move_successors + switch_successors
        np.random.shuffle(all_moves)
        return all_moves

    def _possible_move_actions(self, state):
        """
        return the successive states of the given state, achieved by MOVE action: moving a table to some other
        available table in the restaurant

        :param state: dict[Reservation,list[Table]], representing the current assignment
        :return: list[dict[Reservation,list[Table]]], all possible successive states
        """
        move_successors = []
        scheduler_copy = self.tgm.scheduler.copy()
        for res in self.csp.variables:
            for table in self.csp.domains[res]:
                if self.csp.is_consistent(res, table):
                    state_copy = state.copy()
                    self.tgm.revert_scheduler_assignment(res, scheduler_copy)
                    tables_group = self.tgm.update_scheduler(table, res, scheduler_copy)
                    state_copy.update({res: tables_group})
                    move_successors.append(state_copy)
                    # self.tgm.set_scheduler(scheduler_copy)
        return move_successors

    def _possible_switch_actions(self, state):
        """
        return the successive states of the given state, achieved by SWITCH action: switch between assignments
        of two reservations

        :param state: dict[Reservation,list[Table]], representing the current assignment
        :return: list[dict[Reservation,list[Table]]], all possible successive states
        """
        switch_successors = []
        switch_pairs = itertools.combinations(self.csp.variables, r=2)
        # save a copy of the current scheduler in order to evaluate all possible switch actions from the current
        scheduler_copy = self.tgm.scheduler.copy()
        for res1, res2 in switch_pairs:
            table1 = state[res1][0]
            table2 = state[res2][0]
            # check if the switch is even plausible, and if it is not the same table!
            if table1 in self.csp.domains[res2] and table2 in self.csp.domains[res1] and table1 != table2:
                self.tgm.revert_scheduler_assignment(res1)
                self.tgm.revert_scheduler_assignment(res2)
                # check if the switch is consistent
                if self.csp.is_consistent(res1, table2) and self.csp.is_consistent(res2, table1):
                    state_copy = state.copy()
                    # get new tables assignments and generate the successor state and scheduler
                    new_tables1 = self.tgm.update_scheduler(table2, res1)
                    new_tables2 = self.tgm.update_scheduler(table1, res2)
                    state_copy.update({res1: new_tables1})
                    state_copy.update({res2: new_tables2})
                    switch_successors.append(state_copy)
            self.tgm.set_scheduler(scheduler_copy)
        return switch_successors

    def get_value(self, state: dict):
        """
        :param state: list[dict[Reservation,list[Table]]], assignment that represents a state in the search problem
        :return: float, the value of the objective function over the given state
        """
        # score for chairs utilization
        chairs_score = get_chair_util_score(state, self.tgm) * 100
        # score for satisfying soft constraints
        soft_score = get_soft_tags_score(state, self.soft_tags_amount) * 100
        # score for grouping tables in the same zone and avoiding using not needed zones
        zone_score = get_zone_score(state, self.tgm) * 100
        # get tables utilization penalty (negative score)
        if self.max_time_holes == 0:
            tables_score = 100
        else:
            time_holes_penalty = get_schedule_holes(np.asarray(self.tgm.generate_scheduler(state)))
            tables_score = (1 - (time_holes_penalty / self.max_time_holes)) * 100
        # assert set(state.keys()) == set(self.csp.variables)
        # calculate the date rank ratio score
        score = 0
        for res, tables in state.items():
            # calculate reservation creation date vs table rank.
            # normalize by today's timestamp so we can work with small numbers
            if self.max_date > self.min_date:
                date_value = ((res.created.timestamp() - self.min_date) / (self.max_date - self.min_date)) * 4
            else:
                date_value = 0
            # assert 0 <= date_value <= 4
            # assert 0 <= date_value <= 10
            score += tables[0].rank / (date_value + 1)
        # score for assigning older reservation to more "attractive" tables
        date_rank_score = (score / self.max_date_rank_score) * 100
        # assert 0 <= soft_score <= 100
        # assert 0 <= date_rank_score <= 100
        # assert 0 <= zone_score <= 100
        # assert 0 <= chairs_score <= 100
        # assert 0 <= tables_score <= 100
        return 0.5 * zone_score + 0.1 * soft_score + 0.1 * date_rank_score + 0.2 * chairs_score + 0.1 * tables_score

    def simulated_annealing(self, start_state, T_init, n, cooling_method: callable):
        """
        implementation of the Simulated Annealing algorithm
        :param start_state: list[dict[Reservation,list[Table]]], representing the start state from which we start the
            search
        :param T_init: float, represents the initial temperature
        :param n: int, number of iteration. the algorithm will run for n iterations
        :param cooling_method: callable, cooling schedule, which is a function of step t, that determines the current
            temperature T_t: the temperature at time t
        :return: list[dict[Reservation,list[Table]]], the optimized solution for the problem
        """
        current_state = start_state
        temperature_func = cooling_method(T_init)
        for t in range(n):
            T = temperature_func(t)
            if T == 0:
                return current_state
            else:
                successors = self.get_successors(current_state)
                next_state = np.random.choice(successors)
                next_val = self.get_value(next_state)
                cur_val = self.get_value(current_state)
                delta = next_val - cur_val
                print(f'temperature is {T}, step: {t + 1}/{n} ', end='')
                if delta > 0:
                    current_state = next_state
                    print()
                elif np.random.uniform() < np.exp(delta / T):
                    current_state = next_state
                    print(f'(possibly bad state was chosen)')
                else:
                    print('(state was not changed)')
                # update the tgm agent to be consistent with the current state
                self.tgm.set_scheduler(self.tgm.generate_scheduler(current_state))
        return current_state

    def hill_climbing(self, start_state, n):
        """
        implementation of the Hill Climbing algorithm
        :param start_state: list[dict[Reservation,list[Table]]], representing the start state from which we start the
            search
        :param n: int, number of iteration. the algorithm will run for n iterations
        :return: list[dict[Reservation,list[Table]]], the optimized solution for the problem
        """
        current_state = start_state
        for i in range(n):
            print(f'doing step {i + 1}/{n} (ascending)')
            successors = self.get_successors(current_state)
            successors_values = [self.get_value(s) for s in successors]
            max_idx = int(np.argmax(successors_values))
            next_state = successors[max_idx]
            # if current state is a local maximum, return it
            if successors_values[max_idx] <= self.get_value(current_state):
                return current_state
            # keep climbing
            else:
                current_state = next_state
        return current_state


def get_chair_util_score(state, tgm):
    """
    helper for the get_value method of the local search problem.
    computes the chair utilization score: ration between used chairs and all reserved chairs.

    :param state: list[dict[Reservation,list[Table]]], the current state
    :param tgm: TableGuestManager, the manager of the reservations and tables in the given state
    :return: score: float, in the range [0,1]
    """
    used_chairs = 0  # all chairs that are used by people in the reservations
    total_reserved_chairs = 0  # all chairs of tables that are assigned for some reservation
    for res, tables in state.items():
        total_reserved_chairs += np.sum([table.size for table in tables])
        used_chairs += res.people
    if total_reserved_chairs == 0:
        return 1  # max score (ratio)
    return used_chairs / total_reserved_chairs


def get_soft_tags_score(state, total_soft):
    """
    helper for the get_value method of the local search problem.
    computes the soft tags score: ration between all satisfied soft tags and all soft tags

    :param total_soft: int, total number of soft tags constraints over the reservations
    :param state: list[dict[Reservation,list[Table]]], the current state
    :return: score: float, in the range [0,1]
    """
    if total_soft == 0:
        return 1
    total_satisfied = 0
    for res, tables in state.items():
        for tag in res.tags:
            # we prefer to seat reservation with elder people in the first floor
            if tag == Tags.ELDER and tables[0].floor == 1:
                total_satisfied += 1
            # we prefer to seat reservation with smokers in a smoking area
            if tag == Tags.SMOKING and tag in tables[0].tags:
                total_satisfied += 1
    return total_satisfied / total_soft


def get_schedule_holes(capacity_arr):
    """
    helper for the get_value method of the local search problem.
    count the total number of "schedule-holes"
    :param capacity_arr: ndarray, represents the TGM's scheduler
    :return: int, number of schedule-holes
    """
    # count the number of "holes" in each table schedule. in other words, count the un-utilized periods of each
    # table. that is, the period of time a given cannot be utilized by a reservation. we consider this kind of
    # period as a time in the range of minutes: [15,75].
    # les then 15 minuets time means tight assignment, which is good (in a matter of profit).
    capacity_arr = np.asarray(capacity_arr)
    tables_utilization_penalty = 0  # penalty for un-utilized periods of time
    for row in capacity_arr:
        count = 0
        for rid in row:
            if rid == FREE:
                count += 1
            else:
                if 0 < count < 6:
                    tables_utilization_penalty += 1
                count = 0
    return tables_utilization_penalty


def get_zone_score(state, tgm: TableGuestManager):
    """
    helper for the get_value method of the local search problem.
    computes the "zone-clustering" score: the percentage of people that have been seated in some zone, of the sum
    of all people from all reservations

    :param state: list[dict[Reservation,list[Table]]], the current state
    :param tgm: TableGuestManager, the manager of the reservations and tables in the given state
    :return: score: float, in the range [0,1]
    """
    # calculate zone score, with exponential decreasing rate in the zone number
    # lower zone number gives better score. this means that lower indexed zones are more "attractive" and we want
    # to seat reservations there as much as possible.
    # we assume that a restaurant prefer to fill up a given zone before populating the next one.
    max_zone_score = tgm.get_reservation_amount()
    zone_score = 0
    if max_zone_score == 0:
        return 1  # max score (ratio)
    for res, tables in state.items():
        zone_score += np.exp(-0.99 * (tables[0].zone - 1))
    return zone_score / max_zone_score


def logarithmic_cooling(T):
    """
    a logarithmic cooling schedule that determines the temperature T at time t
    :param T: float, initial temperature
    :return: T_t: float, temperature at time t
    """

    def inner(t):
        return T / (np.log(np.abs(t) + 1) + 1)

    return inner


def geometric_cooling(T):
    """
    a geometric cooling schedule that determines the temperature T at time t
    :param T: float, initial temperature
    :return: T_t: float, temperature at time t
    """

    def inner(t):
        return T * np.power(0.99, t)

    return inner


def exponential_cooling(T):
    """
    an exponential cooling schedule that determines the temperature T at time t
    :param T: float, initial temperature
    :return: T_t: float, temperature at time t
    """

    def inner(t):
        return T * np.exp(-0.99 * t)

    return inner


def init_agents(table_file, res_file):
    """
    initializes the TGM and CSP agents for the local search problem
    :param table_file: str, csv file containing the restaurant's tables records
    :param res_file: str, csv file containing the restaurant's reservations records
    :return: tgm, csp: TableGuestManager, CSP, agents for the local search problem
    """
    # set tgm agent
    tgm = TableGuestManager()
    tgm.read_tables(table_file)
    tgm.read_reservations(res_file)
    tgm.init_scheduler()
    # set csp agent
    relations = [size_constraint, availability_constraint, tags_constraint, area_constraint]
    csp = CSP(tgm.reservations, tgm.tables, tgm)
    csp.init_constraints(relations)
    return tgm, csp


def init_search_parser():
    """
    initialize the parser to parse the user arguments
    :return: parser: ArgumentParser instance
    """
    algorithms = ['csp', 'sa', 'hill']
    cooling_schedules = ['log', 'exp', 'geo']
    parser = argparse.ArgumentParser(description='Table Guest Management: Local Search (AI project 67842). '
                                                 'assigning reservations to restaurant tables. the program outputs '
                                                 'readable solution (as .csv file) and solution instance (as .pkl file) '
                                                 'for later evaluation.')
    parser.add_argument('tables', type=str, help='tables csv filename')
    parser.add_argument('reservations', type=str, help='reservations csv filename')
    parser.add_argument('-a', '--algorithm', type=str, help='algorithm to use to solve the problem',
                        default='csp', choices=algorithms)
    parser.add_argument('-T', '--temperature', type=int, help='initial temperature (positive integer)', default=10)
    parser.add_argument('-n', '--n_iterations', type=int,
                        help='number of iterations (non-negative integer)', default=10)
    parser.add_argument('-c', '--cooling', type=str, help='cooling methods to use in SA local search',
                        choices=cooling_schedules, default='geo')
    return parser


def main():
    """
    main method that parses the user arguments and runs the local search problem.
    the method generates files accordingly: pickled (pkl) solution file, and csv file representing the solution for
    the problem.
    :return:
    """
    # init parser and parse the given args
    parser = init_search_parser()
    args = parser.parse_args()
    T = args.temperature
    n = args.n_iterations
    if T <= 0:
        raise argparse.ArgumentTypeError(f'value {T} for T is invalid. only positive integers are accepted')
    elif n < 0:
        raise argparse.ArgumentTypeError(f'value {n} for n is invalid. only non-negative integers are accepted')
    tables = parse_filename(args.tables)
    reservations = parse_filename(args.reservations)
    if args.cooling == 'exp':
        cooling_method = exponential_cooling
    elif args.cooling == 'log':
        cooling_method = logarithmic_cooling
    else:
        cooling_method = geometric_cooling
    # init agents and start the search
    tgm, csp = init_agents(args.tables, args.reservations)
    ls_agent = TGMLocalSearch(csp, tgm)
    csp_sol = csp.backtracking_search(dict())
    if csp_sol is None:
        print('CSP agent did not find a solution.')
        return
    print(f'CSP agent found a solution!!')
    # csp solution optimization
    if args.algorithm == 'sa':
        cooling_name = cooling_method.__name__.split("_")[0][:3]
        filename = f'solution_{reservations}_sa_{cooling_name}_T={T}_n={n}.pkl'
        print(f'now optimizing search using SA with T={T} n={n} cooling={cooling_name}')
        output_sol = ls_agent.simulated_annealing(csp_sol, T, n, cooling_method)
        data = (output_sol, tgm, f'sa_{cooling_name}')
        pickle_data(data, filename)
    elif args.algorithm == 'hill':
        filename = f'solution_{reservations}_hill_n={n}.pkl'
        print(f'now optimizing search using Hill Climbing with n={n}')
        output_sol = ls_agent.hill_climbing(csp_sol, n)
        data = (output_sol, tgm, 'hill')
        pickle_data(data, filename)
    else:
        filename = f'solution_{reservations}_csp.pkl'
        output_sol = csp_sol
        data = (csp_sol, tgm, 'csp')
        pickle_data(data, filename)
    save_solution_as_csv(output_sol, tgm, reservations)
    print('------------\n'
          'assignment of given reservations was saved as pkl file (pickled files).')
    print('reservation file with its corresponding assignment and TGM scheduler were saved as csv files')


if __name__ == '__main__':
    main()
