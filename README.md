# Table Guest Management using CSP
## Description
this is a readme file giving instruction for using the TableGuestManagement class.
specifically, for explaining the right format of the csv files.
## Instructions (reading Tables)
reading tables from csv file representing the restaurant tables. in order to correctly read a table csv file, the file
must have to following columns, with their respective type:
 
1. `tid` (int), positive integer representing the table id
2. `area` (str), representing the area the corresponding table is located in. options: ['inside', 'outside']
3. `type` (str), type of the table. 
    options: ['normal','tall']
4. `join_group` (str), tid's of other tables separated by ",",the table can be join with. 
    tid's of tables that does not exist(there is no record for them in the given csv file) will be ignored.
5. `floor` (int), floor of the table. positive integer.
6. `rank` (int), rank of the table. positive integer in [1,5]
7. `zone` (int), zone of the table. positive integer.
8. `tags` (str), special tags (attributes) of the table separated by ",".
    options: ['smoking'] indicating the table is in smoking area.
    
### notes: 
* records missing tid or size will be ignored. same for illegal values.
* records with repeated tid of already parsed table with the same tid will be ignored.
* other attributes with missing or illegal values will be assigned with default values.
* see `./example_files/tables_example.csv` for example

## Instructions (reading Reservations)    
reading reservations from csv file. in order to correctly read a reservation csv file, the file must have to following columns,
with their respective type:

1. `people` (int), representing the number of guest of the reservation
2. `created` (str), representing the created date and time of the reservation.
    possible formats: "HH:MM DD/MM/YYYY", "HH:MM DD-MM-YYYY", "DD/MM/YYYY HH:MM", "DD-MM-YYYY HH:MM"
3. `date_time` (str), representing the start date and of the reservation.
    compatible with same format as "created".
4. `tags` (str), special tags (attributes) of the reservation separated by ",".
    options: ['smoking','babychair','wheelchair','elder']. 
5. `area` (str), representing the area the corresponding table is located in. options: ['inside', 'outside']
6. `name` (str), name of the reservation.
    
### notes:

* records with missing or illegal people, date_time values will be ignored.
* all records must have the same "date_time" date (time could be different).
* the first record with valid "date_time" value, will determine the TGM day schedule.
* other attributes with missing or illegal values will be assigned with default values.
* the reservation id (rid) is determined by the row index of the corresponding record in the csv file.
* see `./example_files/reservations_example.csv` for example
        
## Instructions (Reservation time)
* party of `3>=people` will have the table(s) for at most 90 minuets.
* party of `3<people` will have the tables(s) for at most 120 minuets.
* the TGM takes reservation (start time) from `10:00` to `23:00`, in steps of 15 minutes.
* the latest reservation end time is `00:00`. 
* reservation `date_time` time not in steps of 15 minuets will be floored to the closest 15 minutes step.
* please note that depending on the reservation `date_time`, a reservation could not get its full reservation time.
    for example: party of 4 booked at `23:00`.
## Usage
run local_search.py with -h flag for more information

    $ python3 local_search.py -h
    
run result_evaluation.py with -h flag for more information

    $ python3 result_evaluation.py -h 
    
    
## Examples
* see `./example_files` for example csv files: tables, reservations and solution.
* the file `solution_output_example.csv` demonstrates the output csv file, generated from running local_search.py over `tables_example.csv`, `reservations_example.csv`.
* the file `solution_input_example.csv` demonstrates the format csv file of a manual solution as an input for result_evaluation.py with the flag `-m`. 

## Dataset
csv files in `./dataset` were used to generate the figures and solutions evaluations.
