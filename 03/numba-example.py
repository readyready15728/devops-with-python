from functools import wraps
from time import time
import click
import numba
import numpy as np
import pandas as pd

def real_estate_df():
    df = pd.read_csv('https://raw.githubusercontent.com/noahgift/real_estate_ml/master/data/Zip_Zhvi_SingleFamilyResidence.csv')
    df.rename(columns={'RegionName':'ZipCode'}, inplace=True)
    df['ZipCode']=df['ZipCode'].map(lambda x: '{:.0f}'.format(x))
    df['RegionID']=df['RegionID'].map(lambda x: '{:.0f}'.format(x))

    return df

def numerical_real_estate_array(df):
    columns_to_drop = ['RegionID', 'ZipCode', 'City', 'State', 'Metro', 'CountyName']
    df_numerical = df.dropna()
    df_numerical = df_numerical.drop(columns_to_drop, axis=1)
    return df_numerical.values

def real_estate_array():
    df = real_estate_df()
    rea = numerical_real_estate_array(df)
    return np.float32(rea)

def timing(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        t_s = time()
        result = f(*args, **kwargs)
        t_e = time()

        print(f'function: {f.__name__}, args: [{args}, {kwargs}] took: {t_e - t_s} seconds')

        return result
    return wrap

@timing
def expmean(rea, iterations=100000):
    for _ in range(iterations):
        val = rea.mean() ** 2

    return val

@timing
@numba.jit(nopython=True)
def expmean_jit(rea, iterations=100000):
    for _ in range(iterations):
        val = rea.mean() ** 2

    return val

@click.group()
def cli():
    pass

@cli.command()
@click.option('--jit/--no-jit', default=False)
def jit_test(jit):
    rea = real_estate_array()

    if jit:
        click.echo(click.style('Running with JIT', fg='green'))
        expmean_jit(rea)
    else:
        click.echo(click.style('Running NO JIT', fg='red'))
        expmean(rea)

if __name__ == "__main__":
    cli()
