from table_guest_management import *
import pickle
import glob


def save_solution_as_csv(solution: dict, tgm: TableGuestManager, alg_name: str):
    """
    save the given solution as csv file

    :param solution: list[dict[Reservation,list[Table]]], solution for the TGM problem
    :param tgm: TableGuestManager, the manager of the reservations and tables in the given state
    :param alg_name: str, name of the algorithm used to solve the problem
    :return:
    """
    # save csv file with all reservation and a new column: assigned tables
    df = pd.DataFrame(np.zeros(shape=(len(tgm.reservations), 10)),
                      columns=['rid', 'name', 'people', 'area', 'tags', 'created', 'date_time', 'start', 'end',
                               'tables'],
                      dtype=np.int)
    # save in dataframe by creation order
    for i in df.index:
        df.loc[i, 'rid'] = int(tgm.reservations[i].rid)
        df.loc[i, 'name'] = tgm.reservations[i].name
        df.loc[i, 'people'] = int(tgm.reservations[i].people)
        df.loc[i, 'area'] = tgm.reservations[i].area.name
        tags = ''
        for tag in [tag.name for tag in tgm.reservations[i].tags]:
            tags += f'{tag},'
        df.loc[i, 'tags'] = tags
        df.loc[i, 'created'] = str(tgm.reservations[i].created.strftime('%H:%M %d/%m/%Y'))
        df.loc[i, 'date_time'] = str(tgm.reservations[i].date_time.strftime('%H:%M %d/%m/%Y'))
        df.loc[i, 'start'] = tgm.reservations[i].start
        df.loc[i, 'end'] = tgm.reservations[i].end
        tables = ''
        for tid in [table.tid for table in solution[tgm.reservations[i]]]:
            tables += f'{tid},'
        df.loc[i, 'tables'] = tables
    df.to_csv(f'solution_{alg_name}.csv')
    scheduler = tgm.scheduler.copy()
    scheduler.loc[:, :] = np.where(scheduler == -1, '', scheduler)
    scheduler.to_csv(f'scheduler_{alg_name}.csv')


def parse_tabit_reservations(filename, dataframe=None):
    """
    this is a helper tool for parsing the downloaded csv file from the reservation system database
    :param filename: str, reservation file to parse
    :param dataframe: DataFrame, dataframe to parse instead of the csv file

    :return: DataFrame, consistent with the TableGuestManager input format
    """
    # if no dataframe instance was given, the given "filename" is a csv file to read and parse
    if dataframe is None:
        df = pd.read_csv(filename)
        df_name = parse_filename(filename)
    # the "filename" is the name of the file without suffix,
    # and we only need to parse the given dataframe (without read)
    else:
        df = dataframe
        df_name = filename
    # df.name.name = df_name
    df_new = df.rename(columns={'נוצר בתאריך': 'created',
                                'תאריך ושעה': 'date_time',
                                'שם מלא': 'name',
                                'טלפון': 'phone',
                                'אורחים': 'people',
                                'שולחנות': 'tables'})
    # add relevant columns
    df_new.loc[:, 'area'] = -1
    df_new.loc[:, 'tags'] = -1
    # add corresponding tags and attributes
    regex = re.compile('2[01][1-9]')
    for i in df_new.index:
        tags = ''
        record_tags = str(df_new.loc[i, 'תגיות הזמנה'])
        record_notes = str(df_new.loc[i, 'הערות הזמנה'])
        record_tables = str(df_new.loc[i, 'tables'])
        if regex.match(record_tables):
            df_new.loc[i, 'area'] = 'outside'
        if ('מעשנים' in record_tags and 'לא מעשנים' not in record_tags) \
                or ('מעשנים' in record_notes and 'לא מעשנים' not in record_notes):
            tags += 'smoking,'
        if 'כיסא תינוק' in record_tags or 'כיסא תינוק' in record_notes or 'כסא תינוק' in record_notes:
            tags += 'babychair,'
        if 'כיסא גלגלים' in record_tags or 'כיסא גלגלים' in record_notes or 'כסא גלגלים' in record_notes:
            tags += 'wheelchair,'
        if 'מבוגרים' in record_notes or 'למטה' in record_notes:
            tags += 'elder'
        df_new.loc[i, 'tags'] = tags
    df_new.to_csv(f'{df_name}_parsed.csv')
    return df_new


def extract_manual_solution_from_csv(table_filename, man_sol_filename, save=False):
    """
    extracts manual solution (dict[Reservation,list[Table]]) from csv file.
    the given csv solution file must have column named 'tables' or 'שולחנות'

    :param table_filename: str, path to csv file containing the tables
    :param man_sol_filename: str,
        path to csv file containing the reservations with 'tables' column that specifies the allocated tables for
        the corresponding reservation
    :param save: boolean, if True pickled data is saved, otherwise it's not saved
    :return: data: tuple, (dict, TGM, str),
        * manual solution,
        * TableGuestManager over the given tables and reservations,
        * name of agent,
        respectively
    """
    tgm = TableGuestManager()
    tgm.read_tables(table_filename)
    tgm.read_reservations(man_sol_filename)
    manual_sol_csv = pd.read_csv(man_sol_filename).rename(columns={'שולחנות': 'tables'}).fillna(-1)
    manual_sol = {res: [] for res in tgm.reservations}
    for i in manual_sol_csv.index:
        # the current row index in the file, is an index of a row with ignored reservation (containing illegal value(s)
        # it is not in the tgm reservation list. therefore, it should be ignored.
        if i not in tgm.reservation_dict.keys():
            continue
        # if the given manual solution has no tables, it is not a valid solution, therefore it should be ignored.
        if manual_sol_csv.loc[i, 'tables'] == -1 and tgm.reservation_dict.get(i) is not None:
            bad_res = tgm.reservation_dict[i]
            tgm.reservations.remove(bad_res)
            tgm.reservation_dict.pop(bad_res.rid, None)
            manual_sol.pop(bad_res, None)
            continue
        tables_str = str(manual_sol_csv.loc[i, 'tables']).replace(' ', '').split(',')
        # ignore table that their id does not exist in the given csv file
        tables = [int(float(tid)) for tid in tables_str
                  if str(tid).isdigit() and int(float(tid)) in tgm.table_dict.keys()]
        for tid in tables:
            manual_sol[tgm.reservation_dict[i]].append(tgm.table_dict[tid])
    data = (manual_sol, tgm, 'manual')
    if save:
        pickle_data(data, f'solution_manual_{parse_filename(man_sol_filename)}.pkl')
    return data


def parse_filename(filename):
    """
    parses the path, and extracts the name of the file
    :param filename: str, path to a file
    :return: str, the extracted file name
    """
    # get path/filename without its extension
    temp_split = str(filename).rsplit('.', maxsplit=1)
    if len(temp_split) >= 2:
        temp_split = temp_split[-2]
    elif len(temp_split) == 1:
        temp_split = temp_split[0]
    # extract filename (ignore directories)
    temp_name = temp_split.split('/')[-1]
    return temp_name.split('\\')[-1]


def extract_online_reservations(filename):
    """
    parses the given csv filename, and extracts the online reservation
    :param filename: str, path to csv file
    :return: df: DataFrame, consistent with the TableGuestManager input format
    """
    df = pd.read_csv(filename).rename(columns={'נוצר על-ידי': 'created_by'})
    df_online = df[df.loc[:, 'created_by'] == 'הלקוח']
    name_online = f'{parse_filename(filename)}_online'
    return parse_tabit_reservations(name_online, df_online)


def read_all_csv_from_dir(dir_name, files):
    """
    read all csv files from the given directory path
    :param dir_name: str, path to a directory
    :param files: str, extension of files to read
    :return: files_iter: iterator of the corresponding files
    """
    return glob.glob(f'{dir_name}/{files}')


def pickle_data(data, filename):
    """
    save a pickled version of the data
    :param data: object, data to save
    :param filename: str, name of the pickled file
    :return:
    """
    with open(filename, 'bw') as file:
        pickle.dump(data, file)


def read_pickled_data(filename):
    """
    read object from pickled file
    :param filename: pickled file to read from the object data
    :return: object, data in given filename
    """
    with open(filename, 'br') as file:
        data = pickle.load(file)
    return data

