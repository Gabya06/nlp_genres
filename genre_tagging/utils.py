import io

import jellyfish
import numpy as np
import pandas as pd
from statistics import mode
from word2number import w2n


def check_num(x):
    """
    Remove numbers from list
    :param x:list of words
    :return: list of words without any numbers

    E.g: ETL.check_num(['five','four','day','sun']) will return ['day','sun']
    """
    word_list = []
    remove_list = []
    for i in x:
        try:
            w2n.word_to_num(i)
            remove_list.append(i)
        except ValueError:
            word_list.append(i)
    return word_list


def deduplicate_shows(data):
    """
    group by show_name:
    for each show_name group:

        * get the most occuring overview (using token_str)
        * calc how close each token_str is to the mode using levenshtein_distance
        * assign token_str as most freq one and dedupe:
            sort by num of genres and distance keeping the first w max genres
    """

    df_list = []
    for grp, item in data.groupby("show_name"):
        # how close is each token_str to the most freq token_str
        item["distance"] = item["token_str"].apply(
            lambda x: jellyfish.levenshtein_distance(x, mode(item["token_str"]))
        )
        item["token_str"] = mode(item["token_str"])
        temp = (
            item.sort_values(by=["n_genres", "distance"], ascending=False)
            .drop_duplicates(subset=["token_str"], keep="first")
            .reset_index(drop=True)
        )
        df_list.append(temp)
    df_new = pd.concat(df_list)
    # re-index data
    df_new.index = range(0, df_new.shape[0])
    data = df_new


def encode_genres(genre_str_list, lookup_inv_dict):
    """
    Return array with encoded numeric genres based
        on str values & lookup dict {'crime':1}
    """
    encoded_genres = np.array([lookup_inv_dict[y] for y in genre_str_list])
    return encoded_genres


def s3_upload(session, **params):
    """
    Function to upload data to S3
    Uses a dictionary params with keys for data to upload and S3 partition information

    :param session: Athena session
    :param params: dict with keys: data, year, month, week_num
    :return: Uploads data to S3 location based on params dict
    """
    # S3 data to load and path
    df_load = params["data"]
    partition_year = params["year"]
    partition_month = params["month"]
    partition_day = params["day"]
    # filename
    genre_file_name = (
        str(partition_year)
        + str(partition_month).zfill(2)
        + str(partition_day).zfill(2)
    )
    genre_file_name += "_genre_predictions.csv"
    print(genre_file_name)
    # create key for S3
    s3_key = params["path"]
    s3_key += "/".join(
        (
            str("year=" + partition_year),
            str("month=" + partition_month).zfill(2),
            str("day=" + partition_day),
        )
    )
    s3_key += "/" + genre_file_name

    try:
        buffer = io.StringIO()
        df_load.to_csv(buffer, index=False)
        print(f"Key: {s3_key}")
        # put cvs in S3 location specified
        session.put_object(Body=buffer.getvalue(), Bucket=params["bucket"], Key=s3_key)
        print(f"***** Successfully uploaded Genre Predictions to S3: {s3_key} *****")
    except Exception as e:
        print("***** Couldn't load data to S3 *****")
        print(f"Error {e}")
