import pandas as pd

def add_day_index(data: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add an index column to the dataframe based on the date.
    
    Adds a new column 'day' which is a sequential index (0,1,2...) for each unique date
    
    Example:
    date       ticker
    2020-01-01 AAPL
    2020-01-02 AAPL
    2020-01-04 AAPL
    
    day
    0
    1
    2
    
    Args:
        data: pd.DataFrame
        date_col: str

    Returns:
        pd.DataFrame

    """
    # Convert date column to datetime
    data[date_col] = pd.to_datetime(data[date_col])
    
    # Get unique dates and sort them
    unique_dates = data[date_col].sort_values().unique()
    
    # Create a mapping from date to sequential index
    date_to_index = {date: i for i, date in enumerate(unique_dates)}
    
    # Map each date to its sequential index
    data['day'] = data[date_col].map(date_to_index)
    
    return data

if __name__ == "__main__":
    data = pd.DataFrame({
        "date": ["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-01", "2020-01-02", "2020-01-04"],
        "ticker": ["AAPL", "AAPL", "AAPL", "GOOGL", "GOOGL", "GOOGL"]
    })
    data = add_day_index(data, "date")
    print(data)

