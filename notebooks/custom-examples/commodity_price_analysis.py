import yfinance as yf
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os

# Get current directory for output files
current_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(current_dir, 'commodity_prices.log')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)

logger = logging.getLogger(__name__)

def get_commodity_data(symbol, start_date='2005-01-01', end_date='2025-12-31'):
    """Get historical data for a commodity"""
    logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
    try:
        commodity = yf.Ticker(symbol)
        hist = commodity.history(start=start_date, end=end_date)
        
        if not hist.empty:
            logger.info(f"Successfully retrieved {len(hist)} days of data for {symbol}")
            return hist
        else:
            logger.error(f"No data found for {symbol}")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def plot_commodity_prices(gold_data, barley_data):
    """Plot commodity prices"""
    logger.info("Creating price comparison plot")
    
    plt.figure(figsize=(15, 10))
    
    # Plot gold prices
    plt.subplot(2, 1, 1)
    plt.plot(gold_data.index, gold_data['Close'], label='Gold Price', color='gold')
    plt.title('Gold Price History (GC=F)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    
    # Plot barley prices
    plt.subplot(2, 1, 2)
    plt.plot(barley_data.index, barley_data['Close'], label='Barley Price', color='brown')
    plt.title('Barley Price History (ZB=F)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    output_file = os.path.join(current_dir, 'commodity_analysis.png')
    plt.savefig(output_file)
    plt.close()
    logger.info(f"Plot saved to {output_file}")

def main():
    logger.info("="*50)
    logger.info("Starting commodity price analysis")
    
    # Gold futures symbol: GC=F
    # Barley futures symbol: ZB=F
    commodities = {
        'Gold': 'GC=F',
        'Barley': 'ZB=F'
    }
    
    data = {}
    for name, symbol in commodities.items():
        logger.info(f"Processing {name} data")
        commodity_data = get_commodity_data(symbol)
        if commodity_data is not None:
            data[name] = commodity_data
            logger.info(f"{name} data summary:")
            logger.info(f"Average price: ${commodity_data['Close'].mean():.2f}")
            logger.info(f"Maximum price: ${commodity_data['Close'].max():.2f}")
            logger.info(f"Minimum price: ${commodity_data['Close'].min():.2f}")
    
    if len(data) == 2:
        plot_commodity_prices(data['Gold'], data['Barley'])
        logger.info("Analysis completed successfully")
    else:
        logger.error("Could not complete analysis due to missing data")

if __name__ == "__main__":
    main()
