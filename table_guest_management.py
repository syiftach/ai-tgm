import datetime
import enum
import numpy as np
import pandas as pd
from collections import deque
import re
import os
import sys

# valid datetime formats
DATETIME_FORMAT1 = re.compile('\d{2}:\d{2} \d{2}\/\d{2}\/\d{4}')
DATETIME_FORMAT2 = re.compile('\d{2}\/\d{2}\/\d{4} \d{2}:\d{2}')
DATETIME_FORMAT3 = re.compile('\d{2}:\d{2} \d{2}-\d{2}-\d{4}')
DATETIME_FORMAT4 = re.compile('\d{2}-\d{2}-\d{4} \d{2}:\d{2}')

# represents a free slot in the restaurant schedule
FREE = -1



class TableType(enum.Enum):
    """
    this class represents a type of a Table instance
    """
    NORMAL = 1
    TALL = 2

    @staticmethod
    def is_member(member):
        """
        checks if the given string is of a member in TableType
        :param member:
        :return:  TableType instance if represents a string of table type, None otherwise
        """
        if isinstance(member, str):
            return TableType.__members__.get(member.upper())
        else:
            return None


class Area(enum.Enum):
    """
    this class represents an area in a restaurant
    """
    INSIDE = 1
    OUTSIDE = 2

    @staticmethod
    def is_member(member):
        """
        checks if the given string is of a member in Area
        :param member:
        :return:  Area instance if represents a string of some area, None otherwise
        """
        if isinstance(member, str):
            return Area.__members__.get(member.upper())
        else:
            return None


class Tags(enum.Enum):
    """
    this class represents a tag of Table or Reservation instances
    """
    WHEELCHAIR = 1
    BABYCHAIR = 2
    ELDER = 3
    SMOKING = 4

    @staticmethod
    def is_member(member):
        """
        checks if the given string is of a member in Tags
        :param member:
        :return:  Tags instance if represents a string of some Tag, None otherwise
        """
        if isinstance(member, str):
            return Tags.__members__.get(member.upper())
        else:
            return None

    @staticmethod
    def resolve_tags(input_tags):
        """
        parse the given string and return a list of Tags
        :param input_tags: str, represents a series of Tags
        :return: list[Tags]
        """
        tags = []
        temp_tags = str(input_tags).replace(' ', '')
        temp_tags = temp_tags.split(',')
        for tag in temp_tags:
            result = Tags.is_member(tag)
            if result is None:
                continue
            tags.append(result)
        return tags


class Shift(enum.Enum):
    """
    this class represents a shift in a restaurant
    """
    MORNING = 'morning'
    AFTERNOON = 'afternoon'
    EVENING = 'evening'


class Table:
    """
    this class represents a Table in a restaurant
    """

    def __init__(self, tid, size, rank, floor, area, zone, table_type, join_group, tags):
        self.tid = tid  # table id, positive integer
        self.size = size  # number of seats the table has, positive integer
        self.rank = rank  # rank of the table, positive integer
        self.floor = floor  # floor of the area
        self.area = area  # name of the area the table is located in
        self.zone = zone  # int, the zone that the table belongs to
        self.table_type = table_type  # type of the table (see TableType)
        # set(int), group of tid's of tables that can be joined with this table
        self.join_group = {int(t) for t in join_group if t != tid}
        self.tags = tags  # special characteristics this table has. e.g. in smoking area

    def __eq__(self, other):
        return self.tid == other.tid \
               and self.size == other.size \
               and self.floor == other.floor \
               and self.rank == other.rank \
               and self.join_group == other.join_group \
               and self.area == other.area \
               and self.table_type == other.table_type

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return f'Table #{self.tid}'

    def __repr__(self):
        return f'T{self.tid}'

    def __hash__(self):
        return self.tid

    def __gt__(self, other):
        return self.size > other.size

    def __lt__(self, other):
        return self.size < other.size

    def __ge__(self, other):
        return self.size >= other.size

    def __le__(self, other):
        return self.size <= other.size


class Reservation:
    """
    this class represents a reservation in some restaurant
    """

    def __init__(self, rid, name, created, date_time, people, area, tags):
        self.rid = int(rid)  # reservation id
        self.name = name  # name of the reservation
        self.people = int(people)  # number of guest
        self.area = area  # preferred area
        self.tags = tags  # list of special reservation tags
        self.date_time = None  # start time of the reservation
        self.start = None  # str, (HH:MM) starting time of the reservation
        self.end = None  # str, (HH:MM) ending time of the reservation
        self.created = Reservation.format_datetime(created)  # when the reservation was booked

        # adjust the time of the reservation: make sure it follows the correct reservation date-time format
        # minute needs to be in quarters values: (HH:00,HH:15,HH:30,HH:45)
        dt = Reservation.format_datetime(date_time)
        if dt is None:
            dt = datetime.datetime.today().replace(microsecond=0)
        if 0 <= dt.minute <= 14:
            dt = dt.replace(minute=0, second=0)
        elif 15 <= dt.minute <= 29:
            dt = dt.replace(minute=15, second=0)
        elif 30 <= dt.minute <= 44:
            dt = dt.replace(minute=30, second=0)
        elif 45 <= dt.minute <= 59:
            dt = dt.replace(minute=45, second=0)
        # hour needs to be in the range [10,22]
        if dt.hour < 10:
            dt = dt.replace(hour=10, second=0)
        elif dt.hour > 22:
            dt = dt.replace(hour=23, minute=0, second=0)
        self.date_time = dt
        if self.created is None:
            self.created = self.date_time
        self.start = str(self.date_time.time())[:-3]
        end_time = self.set_reservation_end()
        self.end = str(end_time.time().strftime('%H:%M'))

        if self.date_time < self.created:
            self.created = self.date_time
        assert people > 0
        assert isinstance(self.rid, int) and self.rid >= 0
        assert isinstance(tags, list)
        assert self.date_time >= self.created, f'{self.date_time} < {self.created}'

    @staticmethod
    def format_datetime(date_time):
        """
        checks the given date and time of a Reservations
        :param date_time: str or datetime instance, date and time of a Reservation to parse
        :return: datetime instance if given date_time is valid, None otherwise
        """
        try:
            # if given date_time is str instance try to resolve the its format
            if isinstance(date_time, str):
                if DATETIME_FORMAT1.match(date_time) is not None:
                    return datetime.datetime.strptime(f'{date_time}', '%H:%M %d/%m/%Y')
                elif DATETIME_FORMAT2.match(date_time) is not None:
                    return datetime.datetime.strptime(f'{date_time}', '%d/%m/%Y %H:%M')
                elif DATETIME_FORMAT3.match(date_time) is not None:
                    return datetime.datetime.strptime(f'{date_time}', '%H:%M %d-%m-%Y')
                elif DATETIME_FORMAT4.match(date_time) is not None:
                    return datetime.datetime.strptime(f'{date_time}', '%d-%m-%Y %H:%M')
                # inform the user regarding the wrong given format
                else:
                    raise ValueError
            # if the given date_time is datetime instance return it with the following replacements
            elif isinstance(date_time, datetime.datetime):
                return date_time.replace(second=0, microsecond=0)
            # for every other type, return None
            else:
                return None
        except ValueError as e:
            print(f'ERROR: {e}\nexample: 12:00 14/03/2021 or 14/03/2021 12:00', file=sys.stderr)
            return None

    def set_reservation_end(self):
        """
        sets the reservation end time, such that it will not exceed the closing time of the restaurant (which is 24:00)
        :return: datetime, date and time of the Reservation end time
        """
        d, month, y = self.date_time.day, self.date_time.month, self.date_time.year
        h, minute = self.date_time.hour, self.date_time.minute
        # reservation time is 2 hours (max)
        if self.people > 3:
            # if reservation time exceeds the current day, set it to 00:00 of next day (morning)
            if datetime.timedelta(hours=h + 2).days > 0:
                return datetime.datetime(y, month, d + 1)
            else:
                return datetime.datetime(y, month, d, h + 2, minute)
        # reservation time is 1.5 hours (max)
        else:
            # if reservation time exceeds the current day, set it to 00:00 of next day (morning)
            if datetime.timedelta(hours=h + 1, minutes=minute + 30).days > 0:
                return datetime.datetime(y, month, d + 1)
            elif minute + 30 > 59:
                return datetime.datetime(y, month, d, h + 2, (minute + 30) % 30)
            else:
                return datetime.datetime(y, month, d, h + 1, minute + 30)

    def __str__(self):
        return f'Reservation #{self.rid}'

    def __repr__(self):
        return f'R{self.rid}'

    def __hash__(self):
        return hash(str(self.rid) + self.name.upper())

    def __eq__(self, other):
        return self.rid == other.rid

    def __ne__(self, other):
        return not self == other


class TableGuestManager:
    """
    this class represents a Manager of Reservation and Table instances of some restaurant
    """

    def __init__(self):
        # list of Table in the restaurant
        self.tables = []
        # list of Reservation in the restaurant, all of the same day
        self.reservations = []
        # dict[int, Table], mapping from tid to a Table corresponds to the tid
        self.table_dict = dict()
        # dict[int, Reservation], mapping from rid to a Reservation corresponds to the rid
        self.reservation_dict = dict()
        # Table-time scheduler of the TGM
        self.scheduler = None
        # a map dict[zone: {tid_1,...,tid_n}] all tid's are of tables in the corresponding zone
        self.zone_to_tables = dict()
        # mapping from each zone to its total capacity (seats)
        self.capacity = dict()
        # mapping from shift to list[Reservation] that in this shift time range
        # todo (NOTE): not being used in this project
        self.res_by_shift = {shift: [] for shift in Shift}
        # total number of soft tags in reservations
        self.n_soft_tags = 0

    def __eq__(self, other):
        return set(self.tables) == set(other.tables) and set(self.reservations) == set(other.reservations)

    def __ne__(self, other):
        return not self.__eq__(other)

    def _reset_tables(self):
        """
        resets the attributes of the TGM related to Table
        :return:
        """
        self.tables.clear()
        self.table_dict.clear()
        self.scheduler = None
        self.zone_to_tables.clear()
        self.capacity.clear()

    def _reset_reservations(self):
        """
        resets the attributes of TGM related to Reservation
        :return:
        """
        self.reservations.clear()
        self.reservation_dict.clear()
        self.scheduler = None
        for _list in self.res_by_shift.values():
            _list.clear()

    def read_tables(self, filename):
        """
        read and parse a csv file name representing Table records of some restaurant
        :param filename: str, filename of csv file
        :return:
        """
        if not os.path.exists(filename):
            print('Error: read_tables failed. file not found', file=sys.stderr)
            exit(1)
        # reset tgm tables
        self._reset_tables()
        # read tables file
        df_tables = pd.read_csv(filename).fillna(-1)
        try:
            for row in df_tables.itertuples():
                # records with non-positive values will be ignored, as well as repeated tid value
                if row.tid < 0 or row.size <= 0 or row.tid in self.table_dict.keys():
                    continue
                # parse area of the table, default is INSIDE
                result = Area.is_member(row.area)
                if row.area == -1 or result is None:
                    area = Area.INSIDE
                else:
                    area = result
                # parse type of the table, default if NORMAL
                result = TableType.is_member(row.type)
                if row.type == -1 or result is None:
                    table_type = TableType.NORMAL
                else:
                    table_type = result
                # parse the join group of other tables, this table can be joined with
                if row.can_join_with == -1:
                    join_group = set()
                else:
                    join_group = {int(float(n)) for n in str(row.can_join_with).split(',') if n.split('.')[0].isdigit()}
                # parse floor value, default is first floor
                if row.floor <= 0:
                    floor = 1
                else:
                    floor = int(row.floor)
                # parse rank value, default is 2. legal values are integers in [1,5]
                if row.rank <= 0 or row.rank > 5:
                    rank = 2
                else:
                    rank = row.rank
                if row.zone < 0:
                    zone = 1
                else:
                    zone = row.zone
                # parse table attribute tags
                tags = Tags.resolve_tags(row.tags)
                # create a new table
                new_table = Table(row.tid, row.size, rank, floor, area, zone, table_type, join_group, tags)
                self.tables.append(new_table)
                self.table_dict[new_table.tid] = new_table
            # update all tables join groups to be consistent and symmetric
            self.update_tables_join_group()
            # make tables in a join group have consistent attributes
            self.make_tables_join_group_consistent()
            self.update_zones_and_capacity()
        except AttributeError as e:
            attr = str(e).split('\'')[-2]
            print(f'ERROR: Table attribute \'{attr}\' is missing in the given csv file.', file=sys.stderr)
            exit(1)

    def read_reservations(self, filename):
        """
        read and parse a csv file name representing Reservation records of some restaurant
        :param filename: str, filename of csv file
        :return:
        """
        if not os.path.exists(filename):
            print('Error: read_reservations failed. file not found', file=sys.stderr)
            exit(1)
        # reset tgm reservations
        self._reset_reservations()
        scheduler_date = None
        # read reservation file
        df_reservations = pd.read_csv(filename).fillna(-1)
        try:
            # parse reservation columns
            for i, row in enumerate(df_reservations.itertuples()):
                # records with non-positive guests value, or missing reservation start time will be ignored
                resolve_date_time = Reservation.format_datetime(row.date_time)
                if row.people <= 0 or row.date_time == -1 or resolve_date_time is None:
                    continue
                # set the scheduler date to the first seen valid date
                if scheduler_date is None:
                    scheduler_date = resolve_date_time.date()
                # check if the current reservation has different date
                elif resolve_date_time.date() != scheduler_date:
                    continue
                # parse created date
                if row.created == -1:
                    created = None
                else:
                    created = row.created
                # parse tags
                tags = Tags.resolve_tags(row.tags)
                # parse area
                result = Area.is_member(row.area)
                if row.area == -1 or result is None:
                    area = Area.INSIDE
                else:
                    area = result
                # parse name
                if row.name == -1:
                    name = 'NO_NAME'
                else:
                    name = row.name
                new_reservation = Reservation(i, name, created, row.date_time, row.people, area, tags)
                self.reservations.append(new_reservation)
                self.reservation_dict[new_reservation.rid] = new_reservation
            self.assign_reservations_to_shift()
            self.n_soft_tags = np.sum(
                [len(set(res.tags).intersection([Tags.ELDER, Tags.SMOKING])) for res in self.reservations])
        except AttributeError as e:
            attr = str(e).split('\'')[-2]
            print(f'ERROR: Reservation attribute \'{attr}\' is missing in the given csv file.', file=sys.stderr)
            exit(1)

    def init_scheduler(self):
        """
        init the TGM scheduler (DataFrame). the rows represents the Tables in the restaurant, the columns represents
        the hour in a day
        :return:
        """
        cols = [f'{h}:{m}' for h in range(10, 24) for m in ['00', '15', '30', '45']]
        cols.extend(['00:00'])
        rows = sorted([table.tid for table in self.tables])
        self.scheduler = pd.DataFrame(np.full(shape=(len(rows), len(cols)), fill_value=-1), index=rows, columns=cols)

    def set_scheduler(self, new_scheduler):
        """
        assigns the given scheduler to the TGM's scheduler
        :param new_scheduler: DataFrame, new scheduler to assign
        :return:
        """
        self.scheduler = new_scheduler.copy()

    def generate_scheduler(self, assignment):
        """
        generate a TGM scheduler given an assignment
        :param assignment: dict[Reservation:list[Table]],
            assignment to generate the scheduler from
        :return: DataFrame, a tgm scheduler according to the given assignment
        """
        if self.scheduler is None:
            self.init_scheduler()
        df = self.scheduler.copy()
        df.loc[:, :] = FREE
        for res, tables_group in assignment.items():
            for table in tables_group:
                df.loc[table.tid, res.start:res.end][:-1] = res.rid
        return df

    def update_tables_join_group(self):
        """
        update the tables' join group to be consistent and symmetric. a table cannot have a tid of table in its join
        group of table that does not exist in the restaurant. also, if table t1 can be joined with t2, then t2 can be
        joined with t1 (symmetric property)
        :return:
        """
        # update the join groups to be consistent- remove tid's of tables that does not exist
        for table in self.tables:
            temp_tables = []
            for tid in table.join_group:
                # if tid does not exist, ignore it
                if tid in self.table_dict.keys():
                    temp_tables.append(tid)
            table.join_group = set(temp_tables)
        # update join-ability between tables, to preserve its symmetric relation between the tables
        for tid1, table in self.table_dict.items():
            for tid2 in table.join_group:
                self.table_dict[tid2].join_group.add(tid1)

    def make_tables_join_group_consistent(self):
        """
        make all the table in a given join group to be consistent: all are in the same floor, area and zone.
        all have the same rank, TableType and Tags
        :return:
        """
        updated = set()
        for table in self.tables:
            visited = {table.tid}
            table_queue = deque([table])
            while len(table_queue) > 0:
                next_table = table_queue.popleft()
                for tid in next_table.join_group:
                    if tid not in visited:
                        visited.add(tid)
                        table_queue.append(self.table_dict[tid])
            min_tid = min(visited)
            updated.add(min_tid)
            for tid in visited.difference([min_tid]):
                if tid not in updated:
                    updated.add(tid)
                    self.table_dict[tid].area = self.table_dict[min_tid].area
                    self.table_dict[tid].rank = self.table_dict[min_tid].rank
                    self.table_dict[tid].floor = self.table_dict[min_tid].floor
                    self.table_dict[tid].table_type = self.table_dict[min_tid].table_type
                    self.table_dict[tid].tags = self.table_dict[min_tid].tags
                    self.table_dict[tid].zone = self.table_dict[min_tid].zone

    def get_tables_amount(self):
        """
        :return: number of tables that the TGM manages
        """
        return len(self.tables)

    def get_reservation_amount(self):
        """
        :return: number of reservations that the TGM manages
        """
        return len(self.reservations)

    def get_table_max_size(self, table):
        """
        calculate the maximal size that can be achieved by joining all the tables in the join group of the given table
        :param table: Table,
        :return: int, total number of seats of all joint tables
        """
        total_size = 0
        visited = {table.tid}
        table_queue = deque([table])
        while len(table_queue) > 0:
            next_table = table_queue.popleft()
            total_size += next_table.size
            for tid in next_table.join_group:
                if tid not in visited:
                    visited.add(tid)
                    table_queue.append(self.table_dict[tid])
        return total_size

    def is_table_free(self, table, start, end, other_scheduler=None):
        """
        :param table: Table, table of which we check its availability
        :param start: str, represents the start time of the reservation
        :param end: str, represents the end time of the reservation
        :param other_scheduler: DataFrame,
            the TGM instance can apply action over given scheduler instead of its self scheduler
        :return: boolean, True if the table is free in the range [start,end], False otherwise
        """
        if self.scheduler is None and other_scheduler is None:
            return False
        elif self.scheduler is not None and other_scheduler is None:
            return np.alltrue(self.scheduler.loc[table.tid, start:end][:-1] == FREE)
        elif other_scheduler is not None:
            return np.alltrue(other_scheduler.loc[table.tid, start:end][:-1] == FREE)

    def update_scheduler(self, table, reservation, other_scheduler=None):
        """
        sets the reservation to be assigned to the table
        :param reservation: Reservation, reservation to be assigned
        :param table: Table, assignment of the given reservation
        :param other_scheduler: DataFrame,
            the TGM instance can apply action over given scheduler instead of its self scheduler
        :return:
        """
        if self.scheduler is None and other_scheduler is None:
            return
        if other_scheduler is not None:
            current_scheduler = other_scheduler
        else:
            current_scheduler = self.scheduler
        # check if we need to join tables:
        if table.size < reservation.people:
            tables_group = self.join_tables_for_reservation(table, reservation)
            for t in tables_group:
                current_scheduler.loc[t.tid, reservation.start:reservation.end][:-1] = reservation.rid
            # return list of tables that were allocated for the reservation
            return tables_group
        # the given table is large enough, there is no need to join tables
        else:
            current_scheduler.loc[table.tid, reservation.start:reservation.end][:-1] = reservation.rid
            return [table]

    def revert_scheduler_assignment(self, reservation, other_scheduler=None):
        """
        :param reservation: Reservation, a reservation to which we revert the assignment for
        :param other_scheduler: DataFrame,
            the TGM instance can apply action over given scheduler instead of its self scheduler
        :return:
        """
        if reservation.rid not in self.reservation_dict.keys():
            return
        elif self.scheduler is not None and other_scheduler is None:
            self.scheduler.loc[:, :] = np.where(self.scheduler == reservation.rid, FREE, self.scheduler)
        elif other_scheduler is not None:
            other_scheduler.loc[:, :] = np.where(other_scheduler == reservation.rid, FREE, other_scheduler)

    def join_tables_for_reservation(self, table, reservation):
        """
        join tables together (minimal amount that satisfies the reservation people) for the given reservation
        :param table: Table, a table we join with other Tables in its join group in order to have enough seats for the
            given reservation
        :param reservation: Reservation, reservation to be assigned to those tables
        :return: list[Table], list of tables the reservation is assigned to
        """
        target_size = reservation.people
        joined_tables = []  # list of tid's of joined tables
        # the table must be join-able or large enough for the given reservation
        if self.get_table_max_size(table) < target_size:
            return []
        total_size = 0  # size achieved so far
        visited = {table.tid}  # visited "nodes" set
        table_queue = deque([table])  # queue for the next tables to visit
        # continue to join tables together until the joined set is large enough for the reservation
        while total_size < target_size:
            next_table = table_queue.popleft()
            total_size += next_table.size
            joined_tables.append(next_table)
            for tid in next_table.join_group:
                if tid not in visited:
                    visited.add(tid)
                    table_queue.append(self.table_dict[tid])
        return joined_tables

    def update_zones_and_capacity(self):
        """
        updated the zones in some restaurant the TGM represents, and the capacity of each zone
        :return:
        """
        # map each zone to all tables located in that zone
        for table in self.tables:
            if table.zone in self.zone_to_tables.keys():
                self.zone_to_tables[table.zone].add(table.tid)
            else:
                self.zone_to_tables[table.zone] = {table.tid}
        # map each zone to the total capacity in that zone. that is, the number of seats
        for zone, zone_tables in self.zone_to_tables.items():
            self.capacity[zone] = np.sum([self.table_dict[tid].size for tid in zone_tables])

    # todo (NOTE): this functionality is not used in this project, but could be used for other implementations
    #   or purposes.
    def assign_reservations_to_shift(self):
        """
        map each shift to a list of Reservations that are booked to this shift
        :return:
        """
        morning_start = datetime.time(10)
        afternoon_start = datetime.time(13)
        evening_start = datetime.time(18)
        end_day = datetime.time(23)

        for res in self.reservations:
            if morning_start <= res.date_time.time() < afternoon_start:
                self.res_by_shift[Shift.MORNING].append(res)
            elif afternoon_start <= res.date_time.time() < evening_start:
                self.res_by_shift[Shift.AFTERNOON].append(res)
            elif evening_start <= res.date_time.time() <= end_day:
                self.res_by_shift[Shift.EVENING].append(res)


# ============================== HELPERS ============================== #


def get_sorted_by_size(reservations, reverse=True):
    """
    sort list of reservation by the number of people
    :param reservations: list[Reservation], list of reservation to sort
    :param reverse: boolean, if True the returned list is in descending order, otherwise in ascending order
    :return: list[Reservation], sorted list of reservation
    """

    def sort_by_size(reservation):
        return reservation.people

    return sorted(reservations, key=sort_by_size, reverse=reverse)


def get_sorted_by_created(reservations, reverse=False):
    """
    sort list of reservation by the date of creation
    :param reservations: list[Reservation], list of reservation to sort
    :param reverse: boolean, if True the returned list is in descending order, otherwise in ascending order
    :return: list[Reservation], sorted list of reservation
    """

    def sort_by_created(reservation):
        return reservation.created

    return sorted(reservations, key=sort_by_created, reverse=reverse)


def get_sorted_by_tags(reservations, reverse=True):
    """
    sort list of reservation by the number of Tags
    :param reservations: list[Reservation], list of reservation to sort
    :param reverse: boolean, if True the returned list is in descending order, otherwise in ascending order
    :return: list[Reservation], sorted list of reservation
    """

    def sort_by_tags(reservation):
        return len(reservation.tags)

    return sorted(reservations, key=sort_by_tags, reverse=reverse)


def get_sorted_by_rid(reservations):
    """
    sort list of reservation by the reservation id (rid)
    :param reservations: list[Reservation], list of reservation to sort
    :param reverse: boolean, if True the returned list is in descending order, otherwise in ascending order
    :return: list[Reservation], sorted list of reservation
    """

    def sort_by_rid(reservation):
        return reservation.rid

    return sorted(reservations, key=sort_by_rid)


if __name__ == '__main__':
    pass
