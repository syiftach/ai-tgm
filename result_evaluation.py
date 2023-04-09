from local_search import *
import matplotlib.pyplot as plt


def get_fig(title, xlabel, ylabel, subplots):
    """
    return Figure Instance with the given paramaters as attribures

    :param title: str, title of the figure
    :param xlabel: str, x axis label
    :param ylabel: str, y axis label
    :param subplots: int, index of subplot
    :return:
        fig: Figure instance
        ax: Axes, corresponding to fig
    """
    fig = plt.figure()
    ax = fig.add_subplot(subplots)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def evaluate_solution(solution, tgm: TableGuestManager):
    """
    evaluated the given solution according to the soft constraints paramaters
    :param solution: list[dict[Reservation,list[Table]]], solution for the TGM problem
    :param tgm: TableGuestManager, the manager of the reservations and tables in the given state
    :return: evaluation: dict[str,float or dict[int,float]], score evaluation for the given solution
    """
    """soft constraints score"""
    # count total soft constraints, and compute the ratio of those satisfied
    soft_score = get_soft_tags_score(solution, tgm.n_soft_tags) * 100
    """chairs score"""
    # count the unused chairs (un-utilized chairs for the corresponding reservation)
    unused_chairs = 0
    for res, tables in solution.items():
        tables_size = [table.size for table in tables]
        unused_chairs += np.abs(np.sum(tables_size) - res.people)
    """table allocation score"""
    # count the "dead" table time periods
    alloc_score = get_schedule_holes(np.asarray(tgm.generate_scheduler(solution)))
    """zone score"""
    zone_usage = {zone: 0 for zone in tgm.zone_to_tables.keys()}
    all_people = np.sum([res.people for res in tgm.reservations])
    for res, tables in solution.items():
        zone_usage[tables[0].zone] += (res.people / all_people) * 100
    """date-rank score"""
    # in order order to calculate the date-rank-score we take all the tables with rank in {4,5}, denote it with r.
    # and all the n reservation and define k=min{r,n}, and then compute the average of the first k reservations relative
    # to their creation date.
    # then, we compute the average rank of the k tables that those reservations are assigned to.
    # the result is the date-rank score.
    best_tables = [table for table in tgm.tables if table.rank > 3]
    res_created = np.asarray([res.created.timestamp() for res in solution.keys()])
    bound = np.min([len(best_tables), res_created.size])
    argsort_created = np.argsort(res_created)
    earliest_created = np.asarray(tgm.reservations)[argsort_created[:bound]]
    k_date_rank_average = np.sum([solution[res][0].rank for res in earliest_created]) / bound

    return {'chairs_score': unused_chairs,
            'soft_score': soft_score,
            'alloc_score': alloc_score,
            'zone_score': zone_usage,
            'date_rank_score': k_date_rank_average}


def calculate_average_evaluation(results_eval):
    """
    calculate the average of the different solutions evaluations
    :param results_eval: list[dict], list of solution evaluations to calculate the average for
    :return: average evaluation: dict[str,float or dict[int,float]], average score evaluation for the given
        list of solution evaluations
    """
    n_results = len(results_eval)
    if n_results <= 1:
        return results_eval
    average_chairs_score = np.sum([eval_dict['chairs_score'] for eval_dict in results_eval]) / n_results
    average_soft_score = np.sum([eval_dict['soft_score'] for eval_dict in results_eval]) / n_results
    average_alloc_score = np.sum([eval_dict['alloc_score'] for eval_dict in results_eval]) / n_results
    average_date_rank_score = np.sum([eval_dict['date_rank_score'] for eval_dict in results_eval]) / n_results
    average_zone_score = {zone: 0 for zone in results_eval[0]['zone_score'].keys()}
    for eval_dict in results_eval:
        for zone, value in eval_dict['zone_score'].items():
            average_zone_score[zone] += (value / n_results)

    return {'chairs_score': average_chairs_score,
            'soft_score': average_soft_score,
            'alloc_score': average_alloc_score,
            'date_rank_score': average_date_rank_score,
            'zone_score': average_zone_score}


def print_results_summery(method_eval_list):
    """
    prints to stdout summery of the evaluations results for the given solution evaluations list
    :param method_eval_list: list[dict], list of solution evaluations to print the summery for
    :return:
    """
    print('*** results summery ***')
    for sol_eval, name in method_eval_list:
        print(f'result for {name.upper()} solution:')
        print(f'number of unused chairs:      {sol_eval["chairs_score"]}\n'
              f'% soft constraints satisfied: {sol_eval["soft_score"]}\n'
              f'number of schedule holes:     {sol_eval["alloc_score"]}\n'
              f'average table rank score:     {sol_eval["date_rank_score"]}')
        for zone in sol_eval['zone_score'].keys():
            print(f'zone-{zone} usage: {sol_eval["zone_score"][zone]}')
        print('-----------------------------')
    print('*** end of results summery ***')


def plot_results_summery(method_eval_list, save=False, description=''):
    """
    plot figures that give a graphical representation of the results
    :param method_eval_list: list[dict], list of solution evaluations to plot figures for
    :param save: boolean, if True will save figures, otherwise won't be saved
    :param description: str, another description for the figure
    :return:
    """
    n_results = len(method_eval_list)
    if n_results < 1:
        return
    print('showing result figures...')
    chairs = 'chairs_score'
    soft = 'soft_score'
    alloc = 'alloc_score'
    zone = 'zone_score'
    date_rank = 'date_rank_score'
    index1 = np.arange(len(method_eval_list[0][0]) - 1)
    index2 = np.arange(len(method_eval_list[0][0][zone].keys()))
    bar_width1 = 0.8 / n_results
    bar_width2 = 0.8 / len(index2)
    fig1, ax1 = get_fig(f'Results 1/2: Soft Constraints {description}', 'category', 'value', 111)
    fig2, ax2 = get_fig(f'Results 2/2: Zones Usage {description}', 'zones', '% people', 111)
    for i, pair in enumerate(method_eval_list):
        method_eval, name = pair
        # plot the results (without zone usage)
        y_arr = [method_eval[chairs], method_eval[soft], method_eval[alloc], method_eval[date_rank]]
        ax1.bar(index1 + (i * bar_width1),
                y_arr,
                label=f'{name}',
                width=bar_width1)
        for j, y in enumerate(y_arr):
            ax1.text(index1[j] + (i * bar_width1), y + 3 + ((-1) ** i), str(np.around(y, decimals=1)),
                     color='black',
                     fontweight='bold')
        # plot the usage rates for each one of the zones
        y_arr = [method_eval[zone][zid] for zid in method_eval[zone].keys()]
        ax2.bar(index2 + (i * bar_width2),
                y_arr,
                label=f'{name}',
                width=bar_width2)
        for j, y in enumerate(y_arr):
            ax2.text(index2[j] + (i * bar_width2), y + 3 * 0 + ((-1) ** i), str(np.around(y, decimals=1)),
                     color='black',
                     fontweight='bold')
    ax1.set_xticks(index1 + bar_width1 / 2)
    ax1.set_xticklabels(['unused chairs', '% soft satisfied', '"schedule-holes"', 'average\ndate-rank\nscore'])
    ax1.legend()
    ax2.set_xticks(index2 + (bar_width2 / 2))
    ax2.set_xticklabels([f'zone-{zid}\nusage' for zid in method_eval_list[0][0][zone].keys()])
    ax2.legend()
    plt.show()
    plt.clf()
    if save:
        fig1.savefig('soft_constraints_comparison.png')
        fig2.savefig('zone_usage_comparison.png')
        print('figures were saved as png files')


def init_eval_parser():
    """
    initialize the parser to parse the user arguments
    :return: parser: ArgumentParser instance
    """
    parser = argparse.ArgumentParser(description='Table Guest Management: Solution(s) Evaluation (AI project 67842). '
                                                 'the program can evaluate and compare multiple solution, '
                                                 'and can also generate .pkl file from manual .csv file solution.')
    parser.add_argument('-s', '--solutions',
                        help='solution(s) to evaluate. must be at least one solution. '
                             'files must be pickled (pkl) generated by local_search.py '
                             'or by extraction from manual solution (by using -m flag)', nargs='+',
                        type=str)
    parser.add_argument('-m', '--manual_sol',
                        help='evaluates manual solution, and saves the result as .pkl file. '
                             'need to specify sol file and then tables file: -m <manual sol> <tables> '
                             '(both .csv files). example: "-m sol.csv table.csv"',
                        type=str, nargs=2)
    parser.add_argument('-f', '--save_figs', help='saves the result figures as png file', action='count', default=0)
    parser.add_argument('-v', '--verbose', help='print results summery to stdout', action='count', default=0)
    return parser


def main():
    """
    main method that parses the user arguments and runs the local search problem.
    the method generates files accordingly: pickled (pkl) solution file, and png file containing the created figures
    :return:
    """
    parser = init_eval_parser()
    args = parser.parse_args()
    if args.save_figs >= 1:
        save = True
    else:
        save = False
    data_tuples = []
    if args.solutions is not None:
        for filename in args.solutions:
            suffix = filename.split('.')[-1]
            if suffix != 'pkl':
                raise argparse.ArgumentTypeError('given solutions must be a .pkl file')
            data = read_pickled_data(filename)
            if not isinstance(data[0], dict) \
                    or not isinstance(data[1], TableGuestManager) \
                    or not isinstance(data[2], str):
                raise Exception('given .pkl file is in wrong format')
            data_tuples.append(data)
    if args.manual_sol is not None:
        manual_sol_csv, tables_manual_csv = args.manual_sol
        if manual_sol_csv.split('.')[-1] != 'csv' or tables_manual_csv.split('.')[-1] != 'csv':
            raise Exception('given manual solution and tables files need to be in csv format')
        data_tuples.append(extract_manual_solution_from_csv(tables_manual_csv, manual_sol_csv))
    # make sure that all solutions corresponding tgm's are equal
    tgm_agents = [sol[1] for sol in data_tuples]
    pairs = itertools.combinations(tgm_agents, r=2)
    for agent1, agent2 in pairs:
        if agent1 != agent2:
            raise Exception('tgm instances are not consistent! (must have the same set of tables and reservations')
    # evaluate solutions
    solutions_evaluation = []
    for sol, agent, name in data_tuples:
        solutions_evaluation.append((evaluate_solution(sol, agent), name))
    # plot results
    plot_results_summery(solutions_evaluation, save=save)
    if args.verbose >=1:
        print_results_summery(solutions_evaluation)


def agents_comparison(path, agents):
    """
    compares different agents

    :param path: str, path to the directory containing all csv files
    :param agents: list[str], list of agents to compare
    :return:
    """
    filename = f'average_{path.replace("/", ".")}'
    # mapping from agent's name to its solution evaluations (list[dict])
    solutions_evals = {agent: [] for agent in agents}
    # mapping from agent's name to its average evaluation dict
    average_evals = {agent: None for agent in agents}
    csv_files = read_all_csv_from_dir(path, '*.csv')
    for i, res_file in enumerate(csv_files):
        print(f'starting csp search for file {res_file}', end='...')
        tgm, csp = init_agents('./tables_karma.csv', res_file)
        csp_sol = csp.backtracking_search(dict())
        ls_agent = TGMLocalSearch(csp, tgm)
        for agent in agents:
            # compute manual/online solutions results
            if agent == 'manual':
                man_sol = extract_manual_solution_from_csv('./tables_karma.csv', res_file)
                solutions_evals['manual'].append(evaluate_solution(man_sol[0], man_sol[1]))
            elif agent == 'online':
                man_sol = extract_manual_solution_from_csv('./tables_karma.csv', res_file)
                solutions_evals['online'].append(evaluate_solution(man_sol[0], man_sol[1]))
            # append csp solution
            elif agent == 'csp' and csp_sol is not None:
                print('ok')
                solutions_evals['csp'].append(evaluate_solution(csp_sol, tgm))
            # optimize csp solution with SA agent
            elif agent == 'sa' and csp_sol is not None:
                print(f'csp found solution for file {i + 1}/{len(csv_files)}, now optimizing with SA')
                sa_sol = ls_agent.simulated_annealing(csp_sol, 100, 1000, exponential_cooling)
                solutions_evals['sa'].append(evaluate_solution(sa_sol, tgm))
            # optimize csp solution with hill climbing agent
            elif agent == 'hill' and csp_sol is not None:
                print(f'csp found solution for file {i + 1}/{len(csv_files)}, now optimizing with HILL')
                hill_sol = ls_agent.hill_climbing(csp_sol, 100)
                solutions_evals['hill'].append(evaluate_solution(hill_sol, tgm))
            elif csp_sol is None:
                print(f'X')
                break
    # calculate the average over all solutions, for each agent
    for agent in agents:
        average_evals[agent] = calculate_average_evaluation(solutions_evals[agent])
        filename += f'_{agent}'
    # pickle data
    filename += '.pkl'
    pickle_data(average_evals, filename)
    # plot the results
    plot_results_summery([(average_evals[agent], agent) for agent in agents])


if __name__ == '__main__':
    main()
