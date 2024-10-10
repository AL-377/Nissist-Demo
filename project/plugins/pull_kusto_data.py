from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
from azure.kusto.data.helpers import dataframe_from_result_table

from taskweaver.plugin import Plugin, register_plugin


@register_plugin
class KustoPullData(Plugin):
    def __call__(
        self,
        kusto_query: str,
    ):
        import pandas as pd

        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", None)

        connection = self.config.get("connection")
        database = self.config.get("database")
        aad_app_id = self.config.get("aad_app_id")
        app_key = self.config.get("app_key")
        authority_id = self.config.get("authority_id")

        kcsb = KustoConnectionStringBuilder.with_aad_application_key_authentication(
            connection_string=connection,
            aad_app_id=aad_app_id,
            app_key=app_key,
            authority_id=authority_id,
        )
        
        client = KustoClient(kcsb)

        response = client.execute(database, kusto_query)
        df = dataframe_from_result_table(response.primary_results[0])

        description = (
            "After executing the query:\n{query}, the returned dataframe "
            "has the shape of {shape} and schema of {schema}"
            "The data example is:\n{data}".format(
                query=kusto_query,
                shape=df.shape,
                schema=df.dtypes,
                data=df.head(1),
            )
        )

        return df, description
