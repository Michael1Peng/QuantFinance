import akshare as ak
import logging
from datetime import datetime
import time
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get current directory for output files
current_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(current_dir, 'ashare_prices.log')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)

logger = logging.getLogger(__name__)

def get_stock_historical_data(symbol, start_date='20150101', end_date='20251231', max_retries=3):
    """获取单个A股股票的历史数据，带重试机制"""
    logger.info(f"开始获取 {symbol} 从 {start_date} 到 {end_date} 的历史数据")
    
    for attempt in range(max_retries):
        try:
            # 使用akshare获取股票数据
            hist = ak.stock_zh_a_hist(symbol=symbol, start_date=start_date, end_date=end_date, adjust="qfq")
            
            if not hist.empty:
                # 获取股票名称
                stock_info = ak.stock_individual_info_em(symbol)
                company_name = stock_info.iloc[0]['value'] if not stock_info.empty else symbol
                
                # 获取市值信息
                market_info = ak.stock_zh_a_spot_em()
                market_cap = market_info[market_info['代码'] == symbol]['总市值'].iloc[0] if not market_info.empty else 0
                
                logger.info(f"成功获取 {symbol} 的历史数据，共 {len(hist)} 个交易日")
                return {
                    'symbol': symbol,
                    'name': company_name,
                    'history': hist,
                    'market_cap': market_cap
                }
            else:
                logger.warning(f"尝试 {attempt + 1}/{max_retries}: {symbol} 数据不完整")
                
        except Exception as e:
            logger.warning(f"尝试 {attempt + 1}/{max_retries}: 错误 - {str(e)}")
        
        if attempt < max_retries - 1:
            time.sleep(2)
    
    return None

def analyze_price_distribution(stock_data, ax1=None, ax2=None):
    """分析股票价格分布"""
    prices = stock_data['history']['收盘']
    
    # 计算基本统计量
    mean_price = prices.mean()
    std_price = prices.std()
    skewness = stats.skew(prices)
    kurtosis = stats.kurtosis(prices)
    
    # 进行正态性检验
    _, p_value = stats.normaltest(prices)
    
    logger.info(f"\n{stock_data['symbol']} - {stock_data['name']} 价格分布分析:")
    logger.info(f"平均价格: ¥{mean_price:.2f}")
    logger.info(f"标准差: ¥{std_price:.2f}")
    logger.info(f"偏度: {skewness:.2f}")
    logger.info(f"峰度: {kurtosis:.2f}")
    logger.info(f"正态性检验 p-value: {p_value:.4f}")
    
    # 如果提供了axes，在指定的subplot上绘图
    if ax1 is not None and ax2 is not None:
        # 子图1：价格历史
        ax1.plot(stock_data['history'].index, prices.values)
        ax1.set_title(f"{stock_data['symbol']} 价格历史")
        ax1.tick_params(axis='x', rotation=45)
        
        # 子图2：分布直方图
        sns.histplot(prices, kde=True, ax=ax2)
        ax2.set_title(f"{stock_data['symbol']} 价格分布")
    
    return {
        'symbol': stock_data['symbol'],
        'name': stock_data['name'],
        'mean': mean_price,
        'std': std_price,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'p_value': p_value,
        'normal_dist': '是' if p_value > 0.05 else '否'
    }

def get_ashare_top_10():
    """获取A股市值前十公司"""
    try:
        # 获取A股实时行情
        stock_df = ak.stock_zh_a_spot_em()
        # 按总市值排序并获取前10只股票
        top_stocks = stock_df.sort_values('总市值', ascending=False).head(10)
        
        logger.info("开始获取A股市值前十公司的历史股票数据")
        logger.info(f"将要获取的公司代码: {', '.join(top_stocks['代码'].tolist())}")
        
        results = []
        for _, row in top_stocks.iterrows():
            symbol = row['代码']
            logger.info(f"正在获取 {symbol} 的历史数据...")
            stock_data = get_stock_historical_data(symbol)
            
            if stock_data:
                results.append(stock_data)
                logger.info(f"成功获取 {symbol} ({stock_data['name']}) 的历史数据")
            else:
                logger.error(f"在多次尝试后仍无法获取 {symbol} 的数据")
        
        return results
    except Exception as e:
        logger.error(f"获取A股前十公司数据时出错: {str(e)}")
        return []

def main():
    logger.info("="*50)
    logger.info("开始执行A股数据分析程序")
    try:
        stocks = get_ashare_top_10()
        
        if not stocks:
            logger.error("未能获取任何股票数据")
            return
            
        print("\n=== A股市值前十公司价格分布分析 ===")
        print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 创建一个大图表
        n_stocks = len(stocks)
        fig = plt.figure(figsize=(15, 5*n_stocks))
        
        # 存储所有统计数据
        stats_list = []
        
        for idx, stock in enumerate(stocks):
            # 为每个股票创建两个子图
            ax1 = plt.subplot(n_stocks, 2, 2*idx + 1)
            ax2 = plt.subplot(n_stocks, 2, 2*idx + 2)
            
            stats = analyze_price_distribution(stock, ax1, ax2)
            stats_list.append(stats)
        
        plt.tight_layout()
        plt.savefig(os.path.join(current_dir, 'ashare_top10_analysis.png'))
        plt.close()
        
        # 创建统计数据表格并输出到日志
        stats_df = pd.DataFrame(stats_list)
        stats_df = stats_df[['symbol', 'name', 'mean', 'std', 'skewness', 'kurtosis', 'p_value', 'normal_dist']]
        logger.info("\n统计数据汇总表：\n" + stats_df.to_string())
        
        logger.info("程序执行完成")
        
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
