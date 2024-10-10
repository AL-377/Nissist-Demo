""" KustoDataProvider."""
# pylint: disable=protected-access

import logging
from datetime import timedelta
from azure.kusto.data import (
    KustoClient,
    KustoConnectionStringBuilder,
    ClientRequestProperties,
)
from azure.kusto.data._models import KustoResultTable
from retrying import retry
import pandas as pd
import os

class KustoDataProvider():
    """ KustoDataProvider."""

    def __init__(self, connection_string, secret, aad_app_id, authority_id, login_type='KEY'):
        self.logger = logging.getLogger(__name__)
        # e.g.: 'endpoint=https://**.**.kusto.windows.net;db=**'
        connection = dict(subString.split("=") for subString in connection_string.split(";"))
        self._connection = connection['endpoint']
        self._db = connection['db']
        if login_type == 'AAD':
            self._client = self._create_kusto_client_aad()
        else:
            self._secret = secret
            self._aad_app_id = aad_app_id
            self._authority_id = authority_id
            self._client = self._create_kusto_client_key()

    @staticmethod
    def dataframe_from_result_table(table):
        """Converts Kusto tables into pandas DataFrame.
        :param azure.kusto.data._models.KustoResultTable table: Table received from the response.
        :return: pandas DataFrame.
        """
        if not table:
            raise ValueError()

        if not isinstance(table, KustoResultTable):
            raise TypeError("Expected KustoResultTable got {}".format(type(table).__name__))

        columns = [col.column_name for col in table.columns]
        frame = pd.DataFrame(table.rows, columns=columns)

        # fix types
        for col in table.columns:
            if col.column_type == "bool":
                frame[col.column_name] = frame[col.column_name].astype(bool)

        frame = frame.round(3)
        return frame

    @retry(stop_max_attempt_number=3)
    def query_dir(self, input):
        """
        The input should be in the ingest api's input.
        """
        query = input
        self.logger.info("enriched query: \n %s", query)
        print("enriched query: \n %s", query)

        properties = ClientRequestProperties()
        properties.application = "**"
        properties.set_option("norequesttimeout", True)

        # pylint: disable=protected-access
        properties.set_option("servertimeout", KustoClient._query_default_timeout)
        response = self._client.execute(
            self._db, query, properties=properties
        )
        res = response.primary_results[0]
        return self.dataframe_from_result_table(res)

    def _create_kusto_client_aad(self):
        KustoClient._query_default_timeout = timedelta(minutes=15, seconds=00)
        kcsb = KustoConnectionStringBuilder.with_az_cli_authentication(self._connection)
        self.logger.info("Create kusto client")
        return KustoClient(kcsb)

    def _create_kusto_client_key(self):
        KustoClient._query_default_timeout = timedelta(minutes=15, seconds=00)
        aad_secret = self._secret
        add_app_id= self._aad_app_id
        authority_id = self._authority_id
        kcsb = KustoConnectionStringBuilder.with_aad_application_key_authentication(
            connection_string = self._connection,
            aad_app_id = add_app_id,
            app_key = aad_secret,
            authority_id= authority_id
        )
        self.logger.info("Create kusto client")
        return KustoClient(kcsb)
