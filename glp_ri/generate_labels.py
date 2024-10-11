import xarray as xr
import glp_ri.data_utils as du
import numpy as np
import json
import argparse

KT_TO_METERS_PER_SECOND = 1.852 / 3.6
RI_24_HR_THRESHOLD_KTS = 30
HRS_TO_SECS = 3600
GOES_SHIPS_MATCHING_TOLERANCE = 3600

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument('-f', '--filename', type=str, default=None,
                              help='output filename')
INPUT_ARG_PARSER.add_argument('-y', '--years', type=int, nargs='+', default=None,
                              help='years to compute')


def label_single_storm(cyclone_id_string, goes_ships_matching_tolerance=GOES_SHIPS_MATCHING_TOLERANCE):
    du.parse_cyclone_id(cyclone_id_string)
    assert type(goes_ships_matching_tolerance) is int, \
        f'goes_ships_matching_tolerance must be int; got {type(goes_ships_matching_tolerance)}'

    fname = du.find_file(du.DATA_DIR, cyclone_id_string)
    ds = xr.open_dataset(fname)

    RI_dict = dict()
    for sat_time in ds.satellite_valid_time_unix_sec:
        time_diffs = ds.ships_valid_time_unix_sec.values - sat_time.values
        if np.abs(time_diffs).min() <= goes_ships_matching_tolerance:

            ships_time = ds.ships_valid_time_unix_sec[np.argmin(np.abs(time_diffs))]
            valid_times = ds.ships_valid_time_unix_sec[(0 <= time_diffs) & (time_diffs <= 24*HRS_TO_SECS)]

            ref_intensity = ds.ships_storm_intensity_m_s01.sel(ships_valid_time_unix_sec=ships_time)
            RI = int(np.any([
                (ds.ships_storm_intensity_m_s01.sel(ships_valid_time_unix_sec=t) - ref_intensity)
                >= RI_24_HR_THRESHOLD_KTS * KT_TO_METERS_PER_SECOND for t in valid_times
            ]))

            RI_dict[sat_time.item()] = RI
        else:
            pass

    if len(RI_dict) == 0:
        raise ValueError(f'Found no valid storm times for {cyclone_id_string}')

    return RI_dict


def label_storms_by_years(years, out_fname, goes_ships_matching_tolerance=GOES_SHIPS_MATCHING_TOLERANCE):
    assert type(years) is list, f'years must be a list; got {type(years)}'
    assert type(out_fname) is str, f'out_fname must be str; got {type(out_fname)}'

    print("Processing cyclones for the following years:")
    for year in years:
        print(year)
    print(f"Saving to {out_fname}")

    tc_list = du.find_cyclones(du.DATA_DIR)

    RI_dict_by_year = dict()

    for tc_id in tc_list:
        if du.parse_cyclone_id(tc_id)[0] not in years:
            continue

        print(f'Computing for {tc_id}')
        try:
            RI_dict_by_year[tc_id] = label_single_storm(tc_id, goes_ships_matching_tolerance)
        except Exception as exc:
            print(f"An error occurred: " + type(exc).__name__ + " - " + exc)

    with open(du.DATA_DIR + '/' + out_fname, 'w') as fn:
        json.dump(RI_dict_by_year, fn, indent=4)

    return None


if __name__ == "__main__":
    args = INPUT_ARG_PARSER.parse_args()
    label_storms_by_years(args.years, args.filename)