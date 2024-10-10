from .utils.kusto.kusto_data_provider import KustoDataProvider
import yaml
import os

with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

if 'KUSTO_SECRET' in os.environ:
    KUSTO_SECRET = os.environ['KUSTO_SECRET']
else:
    KUSTO_SECRET = config['KUSTO_SECRET']

if 'KUSTO_AAD_APP_ID' in os.environ:
    KUSTO_AAD_APP_ID = os.environ['KUSTO_AAD_APP_ID']
else:
    KUSTO_AAD_APP_ID = config['KUSTO_AAD_APP_ID']

if 'KUSTO_AUTHORITY_ID' in os.environ:
    KUSTO_AUTHORITY_ID = os.environ['KUSTO_AUTHORITY_ID']
else:
    KUSTO_AUTHORITY_ID = config['KUSTO_AUTHORITY_ID']

if 'KUSTO_CONNECTION' in os.environ:
    KUSTO_CONNECTION = os.environ['KUSTO_CONNECTION']
else:
    KUSTO_CONNECTION = config['KUSTO_CONNECTION']

if 'KUSTO_LOGIN' in os.environ:
    KUSTO_LOGIN = os.environ['KUSTO_LOGIN']
else:
    KUSTO_LOGIN = config['KUSTO_LOGIN']

try:
    general_kusto_provider = KustoDataProvider(KUSTO_CONNECTION, KUSTO_SECRET, KUSTO_AAD_APP_ID, KUSTO_AUTHORITY_ID, KUSTO_LOGIN)
except Exception as e:
    print(f"Failed to create KustoDataProvider: {e}")
    general_kusto_provider = None
def query_kusto_api(query):
    if general_kusto_provider is None:
        return None
    df = general_kusto_provider.query_dir(query)
    return df

if __name__ == '__main__':   
    pass