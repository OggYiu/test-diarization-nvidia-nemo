"""
Client Trade Search Script

This script provides a function to search for client trades in trades.csv
with flexible date filtering (year, month, day).
"""

import csv
from datetime import datetime
from typing import Optional, List, Dict
import os


def search_client_trades(
    client_code: str,
    year: int,
    month: Optional[int] = None,
    day: Optional[int] = None,
    csv_file: str = "trades.csv"
) -> List[Dict[str, str]]:
    """
    Search for client trades in trades.csv based on client code and flexible date criteria.
    
    Parameters:
    -----------
    client_code : str
        The client's account code (e.g., 'P77197', 'M57509')
    year : int
        The year to search (e.g., 2025)
    month : Optional[int]
        The month to search (1-12). If omitted, returns all records in the year.
    day : Optional[int]
        The day to search (1-31). If omitted, returns all records in the month.
    csv_file : str
        Path to the trades CSV file (default: 'trades.csv')
    
    Returns:
    --------
    List[Dict[str, str]]
        A list of dictionaries, each representing a matching trade record.
    
    Examples:
    ---------
    # Search for all trades by client P77197 in 2025
    results = search_client_trades('P77197', 2025)
    
    # Search for all trades by client P77197 in October 2025
    results = search_client_trades('P77197', 2025, 10)
    
    # Search for all trades by client P77197 on October 9, 2025
    results = search_client_trades('P77197', 2025, 10, 9)
    """
    
    # Validate inputs
    if month is not None and (month < 1 or month > 12):
        raise ValueError("Month must be between 1 and 12")
    
    if day is not None and (day < 1 or day > 31):
        raise ValueError("Day must be between 1 and 31")
    
    if day is not None and month is None:
        raise ValueError("Cannot specify day without specifying month")
    
    # Check if file exists
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    matching_records = []
    
    # Read and search the CSV file
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Check if client code matches
            if row.get('ACCode') != client_code:
                continue
            
            # Parse the OrderTime
            order_time_str = row.get('OrderTime', '')
            if not order_time_str:
                continue
            
            try:
                # Parse datetime (format: "2025-10-09 10:57:32.620")
                order_datetime = datetime.strptime(order_time_str.split('.')[0], '%Y-%m-%d %H:%M:%S')
            except (ValueError, IndexError):
                # Skip if date format is invalid
                continue
            
            # Apply date filters
            if order_datetime.year != year:
                continue
            
            if month is not None and order_datetime.month != month:
                continue
            
            if day is not None and order_datetime.day != day:
                continue
            
            # If all filters pass, add to results
            matching_records.append(row)
    
    # Sort results by OrderTime (chronological order)
    matching_records.sort(key=lambda x: x.get('OrderTime', ''))
    
    return matching_records


def print_trade_summary(records: List[Dict[str, str]], client_code: str, 
                       year: int, month: Optional[int] = None, day: Optional[int] = None):
    """
    Print a formatted summary of the trade records.
    
    Parameters:
    -----------
    records : List[Dict[str, str]]
        List of trade records to display
    client_code : str
        The client code searched for
    year : int
        The year searched
    month : Optional[int]
        The month searched (if applicable)
    day : Optional[int]
        The day searched (if applicable)
    """
    # Build date string
    if day is not None:
        date_str = f"{year}-{month:02d}-{day:02d}"
    elif month is not None:
        date_str = f"{year}-{month:02d}"
    else:
        date_str = str(year)
    
    print(f"\n{'='*80}")
    print(f"Trade Search Results for Client: {client_code} | Date: {date_str}")
    print(f"{'='*80}")
    print(f"Total records found: {len(records)}\n")
    
    if not records:
        print("No matching records found.")
        return
    
    # Print records in a formatted way
    for i, record in enumerate(records, 1):
        print(f"Record #{i}:")
        print(f"  Order No: {record.get('OrderNo', 'N/A')}")
        print(f"  Order Time: {record.get('OrderTime', 'N/A')}")
        print(f"  Security: {record.get('SCTYCode', 'N/A')}")
        print(f"  Side: {record.get('OrderSide', 'N/A')}")
        print(f"  Quantity: {record.get('OrderQty', 'N/A')}")
        print(f"  Price: {record.get('OrderPrice', 'N/A')}")
        print(f"  Status: {record.get('OrderStatus', 'N/A')}")
        print(f"  Done Qty: {record.get('doneQty', 'N/A')}")
        print(f"  Done Price: {record.get('donePrice', 'N/A')}")
        print()


def export_results_to_csv(records: List[Dict[str, str]], output_file: str):
    """
    Export search results to a CSV file.
    
    Parameters:
    -----------
    records : List[Dict[str, str]]
        List of trade records to export
    output_file : str
        Path to the output CSV file
    """
    if not records:
        print("No records to export.")
        return
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    
    print(f"Results exported to: {output_file}")


# Example usage
if __name__ == "__main__":
    # Example 1: Search for all trades by client P77197 in 2025
    # print("Example 1: All trades for client P77197 in 2025")
    # results = search_client_trades('P77197', 2025)
    # print_trade_summary(results, 'P77197', 2025)
    
    # # Example 2: Search for all trades by client P77197 in October 2025
    # print("\nExample 2: All trades for client P77197 in October 2025")
    # results = search_client_trades('P77197', 2025, 10)
    # print_trade_summary(results, 'P77197', 2025, 10)
    
    # Example 3: Search for all trades by client P77197 on October 9, 2025
    print("\nExample 3: All trades for client P77197 on October 9, 2025")
    results = search_client_trades('P77197', 2025, 10, 20)
    print_trade_summary(results, 'P77197', 2025, 10, 20)
    
    # Example 4: Export results to CSV
    # if results:
    #     export_results_to_csv(results, 'search_results.csv')
    
    # # Example 5: Search for another client
    # print("\n" + "="*80)
    # print("\nExample 4: All trades for client M57509 in October 2025")
    # results = search_client_trades('M57509', 2025, 10)
    # print_trade_summary(results, 'M57509', 2025, 10)

