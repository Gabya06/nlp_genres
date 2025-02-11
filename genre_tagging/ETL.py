import io
import time

import boto3
import numpy as np
import pandas as pd
from nltk import RegexpTokenizer
from nltk.corpus import stopwords

from athena_helpers import AthenaS3
from athena_helpers import aws_auth
from athena_helpers.aws_auth import aws_access_key_id, aws_secret_access_key

from nlp_genres.genre_tagging import config, genre_params, utils


class ETL:
    """
    Class used for ETL for data used in model training for CTV show genre predictions (Version1)
    Used for:
    1) Athena query for data  from Inscape & TMBD tables
    2) Clean up show overview
    3) Clean up & combine show genres

    Note: this will likely change once TMDB table changes
    """
    def __init__(self) -> None:
        self.default_params = genre_params.default_params
        self.file_source = ""
        self.query = genre_params.show_query_train
        self.data = pd.DataFrame()
        self.text_col = "overview"
        self.stopwords = stopwords.words("English")
        self.tokenizer = RegexpTokenizer(r"\w+")


    # Pulling Athena S3 data
    def get_s3_connection(self):
        """
        Connect to AWS S3 to be able to pull data
        """
        try:
            s3_resource = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=self.default_params["region"],
            )
            return s3_resource
        except Exception as err:
            msg = "Error in Athena connection: %s" % (str(err))
            print(msg)


    def query_athena(self):
        """
        Query Athena
        Update default_params for the session, print location & return query output
        """

        session = AthenaS3.AthenaS3()
        # change output path
        session.__init__(self.default_params)
        # build athena session
        start_time = time.time()
        session.query = self.query
        # run Athena query
        print(f"Starting to query Athena:\n {self.query}")
        output = session.start_query_request()
        query_time = time.time() - start_time
        query_time = np.round(query_time, 3)
        total_mins = query_time / 60.0
        total_mins = np.round(total_mins, 3)

        print(f"Finished running Athena Query in 
              {query_time} seconds ({total_mins} Total Minutes) ..."
              )

        # Print file location for query result
        print(f"\n Athena Query file location: \n{output['file_location']}")
        self.file_source = "/".join(output["file_location"].split("/")[3:])


    def get_s3_data(self):
        """
        Function to get DataFrame with data based on object output file source (path - where in S3)
        * Drops NA's
        :return: DataFrame
        """
        # Build an s3 session to load the file output file source location
        # set bucket to get data
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_auth.aws_access_key_id,
            aws_secret_access_key=aws_auth.aws_secret_access_key,
        )

        # get S3 data (new shows found) - get last one in list
        df_new_obj = s3_client.get_object(
            Bucket=self.default_params["bucket"], Key=self.file_source
        )
        # read object csv
        df = pd.read_csv(io.BytesIO(df_new_obj["Body"].read()))
        self.data = df


    def get_csv_data(self):
        """
        Read data from csv file
        """
        try:
            self.data = pd.read_csv(self.file_source)
        except FileNotFoundError:
            print("Check file source, file not found.\n Unable to load data")


    def tokenize_text(self):
        """
        Split text from overview column & clean it up
        Remove stopwords
        Return each row as a list of tokens
        """
        # split text
        tokens = self.data[self.text_col].map(lambda x: self.tokenizer.tokenize(x))
        # strip white spaces & lower case
        tokens = tokens.map(lambda x: [i.lower().strip("_") for i in x])
        # remove stop words
        tokens = tokens.map(lambda x: [i for i in x if i not in self.stopwords])
        # remove empty strings
        tokens = tokens.map(lambda x: [i for i in x if i != ""])
        return tokens


    def clean_training_data(self):
        """
        Clean up data :
            tokenize overview column
            drop additional bad data values
            remove number words (eg: one, two...)
        """

        # df = self.data.copy()
        self.data = self.data.drop_duplicates()
        self.data = self.data.show_name.str.strip()
        self.data = self.data[self.data.show_name != " "]

        self.data = self.data[
            (self.data.show_name.duplicated() & self.data.overview.duplicated())
        ]
        # lowercase
        self.data.overview = self.data.overview.str.lower()
        # remove null genres & overview
        self.data = self.data.loc[self.data[self.text_col] != ""]
        self.data = self.data.loc[~self.data[self.text_col].isnull()]
        self.data = self.data.loc[self.data[self.text_col].str.len() > 3]
        self.data = self.data.loc[~self.data[self.text_col].isin([config.values_drop])]
        print("\n...Tokenizing text...")
        # tokenize text description
        self.data = self.data.assign(tokens=self.tokenize_text())
        print("\n...Removing numbers...")
        # remove number words from string
        self.data.tokens = self.data.tokens.map(lambda x: utils.check_num(x))
        # assign column for tokens as str
        self.data = self.data.assign(
            token_str=self.data.tokens.map(lambda x: " ".join(x))
        )

        # re-index
        self.data.index = range(0, self.data.shape[0])


    def clean_genres(self):
        """
        Clean up genres:
        * Make replacements to original genre str
        * Add column for only 1st genre = genre_1
        * Add column for cleaned up & final genres  = genre_2
        """
        dict_replace = {
            "sci-fi & fantasy": "science fiction",
            "mystery": "science fiction",
            "fantasy": "science fiction",
            # combine to action
            "war": "action",
            "action & adventure": "action",
            "adventure": "action",
            # kids
            "kids": "animation",
            # documentary:
            "history": "documentary",
            "war & politics": "documentary",
            # talk
            "talk": "talk-show",
            "soap": "talk-show",
        }
        print("\n...Genre cleaning...")
        #  GENRE CLEANUP
        self.data = self.data.loc[~self.data.genre.isnull()]
        self.data.genre = self.data.genre.str.lower()
        self.data = self.data.assign(
            genre=self.data.genre.map(
                lambda x: "|".join(
                    np.unique(
                        [
                            dict_replace[i] if dict_replace.get(i, None) else i
                            for i in x.split("|")
                        ]
                    )
                )
            )
        )
        self.data.genre = self.data.genre.map(
            lambda x: "|".join(sorted([i.strip() for i in x.split("|")]))
        )
        self.data = self.data.assign(
            genre_split=self.data.genre.map(lambda x: x.split("|"))
        )

        # assign column for number of genres
        self.data = self.data.assign(
            n_genres=self.data.genre_split.map(lambda x: len(x))
        )
        # column for 1st genre in list
        self.data = self.data.assign(
            genre_1=self.data.genre.map(lambda x: x.split("|")[0])
        )
        # drop this
        self.data = self.data.loc[~(self.data.genre_1 == "en")]
        self.data.genre_1 = self.data.genre_1.map(lambda x: x.strip())
        # re-index
        self.data.index = range(0, self.data.shape[0])


    def remove_duplicate_shows(self):
        """Helper function to remove duplicated shows"""
        ix_list = []
        duped_shows = self.data[self.data.show_name.duplicated()].show_name.unique()
        duped_ix = self.data[self.data.show_name.duplicated()].index
        for grp, item in self.data[self.data.show_name.isin(duped_shows)].groupby(
            "show_name"
        ):
            print(grp)
            longest_str = sorted(item.token_str, key=len)[-1]
            ix_list.append(item[item.token_str == longest_str].index[0])
        ix_drop = set(duped_ix).difference(set(ix_list))
        self.data.drop(list(ix_drop), inplace=True)


    def clean_new_data(self):
        """
        Clean new data text_col
        :return: Assign cleaned data to ETL
        """
        self.data = self.data.drop_duplicates()
        # remove nulls & empty strings in overview & drop bad values
        self.data = self.data.loc[self.data[self.text_col].str.len() > 3]
        self.data = self.data.loc[
            ~self.data[self.text_col].str.lower().isin(config.values_drop)
        ]
        # add column for tokens
        self.data = self.data.assign(tokens=self.tokenize_text())
        # remove number words from string
        self.data.tokens = self.data.tokens.map(lambda x: utils.check_num(x))
        self.data = self.data.assign(
            token_str=self.data.tokens.map(lambda x: " ".join(x))
        )
        self.data = self.data[~self.data.token_str.isnull()]
        self.data.index = range(0, self.data.shape[0])
