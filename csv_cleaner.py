import pandas as pd
def clean_csv(csv_file):
    # Clean the data
    df = pd.read_csv(csv_file)
    df = df.dropna()
    df = df.drop_duplicates()
    # drop rows with first column value 'English'
    df = df[df.iloc[:,0] != 'English']
    # # drop rows with just '.' or ',' or '?' in first column
    df = df[df.iloc[:,0] != '.']
    df = df[df.iloc[:,0] != ',']
    df = df[df.iloc[:,0] != '?']
    df = df.reset_index(drop=True)
    return df
def main():
    # Path: main.py
    folder_list = ['en-ha', 'en-ig', 'en-sw', 'en-yo', 'ha-ig', 'ha-sw', 'ha-yo', 'sw-ig', 'yo-ig', 'yo-sw']
    for folder in folder_list:
        dev_path = folder + '/dev.csv'
        df = clean_csv(dev_path)
        df.to_csv(folder+'/cleaned_dev.csv', index=False)
        train_path = folder + '/train.csv'
        df = clean_csv(train_path)
        df.to_csv(folder+'/cleaned_train.csv', index=False)

if __name__ == '__main__':
    main()