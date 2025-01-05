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
log_file = os.path.join(current_dir, 'ashare_gains.log')

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
    logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
    
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
                
                # 数据清洗和异常值处理
                # 1. 确保价格数据为数值型
                hist['收盘'] = pd.to_numeric(hist['收盘'], errors='coerce')
                
                # 2. 移除价格为0或空值的数据
                hist = hist[hist['收盘'] > 0].copy()
                
                # 3. 计算日收益率
                hist['gain'] = hist['收盘'].pct_change() * 100
                
                # 4. 使用 Interquartile Range (IQR) 方法处理异常值
                Q1 = hist['gain'].quantile(0.25)
                Q3 = hist['gain'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 将超出范围的值标记为NaN
                hist.loc[hist['gain'] < lower_bound, 'gain'] = np.nan
                hist.loc[hist['gain'] > upper_bound, 'gain'] = np.nan
                
                logger.info(f"Successfully retrieved {len(hist)} trading days for {symbol}")
                logger.info(f"Valid return data points: {hist['gain'].notna().sum()}")
                
                return {
                    'symbol': symbol,
                    'name': company_name,
                    'history': hist,
                    'market_cap': market_cap
                }
            else:
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: Incomplete data for {symbol}")
                
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries}: Error - {str(e)}")
        
        if attempt < max_retries - 1:
            time.sleep(2)
    
    return None

def analyze_gain_distribution(stock_data, ax1=None, ax2=None):
    """分析股票收益率分布"""
    gains = stock_data['history']['gain'].dropna()  # 删除NaN值
    
    # 计算基本统计量
    mean_gain = gains.mean()
    std_gain = gains.std()
    skewness = stats.skew(gains)
    kurtosis = stats.kurtosis(gains)
    
    # 进行正态性检验
    _, p_value = stats.normaltest(gains)
    
    # 输出统计信息到日志
    logger.info(f"\n{stock_data['symbol']} - {stock_data['name']} Return Analysis:")
    logger.info(f"Mean Daily Return: {mean_gain:.2f}%")
    logger.info(f"Standard Deviation: {std_gain:.2f}%")
    logger.info(f"Skewness: {skewness:.2f}")
    logger.info(f"Kurtosis: {kurtosis:.2f}")
    logger.info(f"Normality Test p-value: {p_value:.4f}")
    logger.info(f"Normal Distribution: {'Yes' if p_value > 0.05 else 'No'}")
    
    # 如果提供了axes，在指定的subplot上绘图
    if ax1 is not None and ax2 is not None:
        # 子图1：收益率时间序列
        ax1.plot(gains.index, gains.values, alpha=0.6, linewidth=1)
        ax1.axhline(y=mean_gain, color='r', linestyle='--', alpha=0.8, label=f'Mean ({mean_gain:.2f}%)')
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        ax1.fill_between(gains.index, mean_gain - std_gain, mean_gain + std_gain, 
                        color='gray', alpha=0.2, label=f'±1 Std ({std_gain:.2f}%)')
        ax1.set_title(f"{stock_data['symbol']} - Daily Returns")
        ax1.set_ylabel('Return (%)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 子图2：分布直方图与正态分布拟合
        sns.histplot(gains, kde=True, ax=ax2, bins=50, alpha=0.6)
        
        # 添加理论正态分布曲线
        xmin, xmax = ax2.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        y = stats.norm.pdf(x, mean_gain, std_gain)
        y = y * (len(gains) * (xmax - xmin) / 50)  # 调整曲线高度以匹配直方图
        ax2.plot(x, y, 'r--', alpha=0.8, label='Normal Distribution')
        
        ax2.set_title(f"Return Distribution (p={p_value:.4f})")
        ax2.set_xlabel('Return (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    return {
        'symbol': stock_data['symbol'],
        'name': stock_data['name'],
        'mean': mean_gain,
        'std': std_gain,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'p_value': p_value,
        'normal_dist': 'Yes' if p_value > 0.05 else 'No'
    }

def get_ashare_top_10():
    """获取A股市值前十公司"""
    try:
        # 获取A股实时行情
        stock_df = ak.stock_zh_a_spot_em()
        # 按总市值排序并获取前10只股票
        top_stocks = stock_df.sort_values('总市值', ascending=False).head(10)
        
        logger.info("Fetching historical data for A-Share top 10 companies")
        logger.info(f"Companies to analyze: {', '.join(top_stocks['代码'].tolist())}")
        
        results = []
        for _, row in top_stocks.iterrows():
            symbol = row['代码']
            stock_data = get_stock_historical_data(symbol)
            
            if stock_data:
                results.append(stock_data)
                logger.info(f"Successfully retrieved data for {symbol} ({stock_data['name']})")
            else:
                logger.error(f"Failed to retrieve data for {symbol} after multiple attempts")
        
        return results
    except Exception as e:
        logger.error(f"Error getting A-Share top 10 companies: {str(e)}")
        return []

def main():
    logger.info("="*50)
    logger.info("Starting A-Share Return Analysis")
    try:
        stocks = get_ashare_top_10()
        
        if not stocks:
            logger.error("No stock data retrieved")
            return
            
        print("\n=== A-Share Top 10 Companies Return Analysis ===")
        print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 创建一个大图表
        n_stocks = len(stocks)
        fig = plt.figure(figsize=(15, 7*n_stocks))
        
        # 存储所有统计数据
        stats_list = []
        
        for idx, stock in enumerate(stocks):
            # 为每个股票创建两个子图
            ax1 = plt.subplot(n_stocks, 2, 2*idx + 1)
            ax2 = plt.subplot(n_stocks, 2, 2*idx + 2)
            
            stats = analyze_gain_distribution(stock, ax1, ax2)
            stats_list.append(stats)
        
        plt.tight_layout()
        output_image = os.path.join(current_dir, 'ashare_top10_gains_analysis.png')
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建统计数据表格并输出到日志
        stats_df = pd.DataFrame(stats_list)
        stats_df = stats_df[['symbol', 'name', 'mean', 'std', 'skewness', 'kurtosis', 'p_value', 'normal_dist']]
        
        # 重命名列以便更好地显示
        stats_df.columns = ['Symbol', 'Name', 'Mean Return(%)', 'Std Dev(%)', 'Skewness', 'Kurtosis', 'P-value', 'Normal Dist']
        
        # 格式化数值列
        stats_df['Mean Return(%)'] = stats_df['Mean Return(%)'].round(2)
        stats_df['Std Dev(%)'] = stats_df['Std Dev(%)'].round(2)
        stats_df['Skewness'] = stats_df['Skewness'].round(2)
        stats_df['Kurtosis'] = stats_df['Kurtosis'].round(2)
        stats_df['P-value'] = stats_df['P-value'].round(4)
        
        # 输出汇总信息
        logger.info("\n=== Return Analysis Summary ===")
        logger.info(f"Sample Size: {len(stats_df)} companies")
        logger.info(f"Companies with Normal Distribution: {(stats_df['Normal Dist'] == 'Yes').sum()}")
        logger.info(f"Companies with Positive Mean Return: {(stats_df['Mean Return(%)'] > 0).sum()}")
        logger.info("\nReturn Statistics Summary Table:")
        logger.info("\n" + stats_df.to_string(index=False))
        
        # 保存统计结果到CSV文件
        output_csv = os.path.join(current_dir, 'ashare_top10_gains_stats.csv')
        stats_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        logger.info(f"\nAnalysis Results Saved:")
        logger.info(f"1. Charts: {output_image}")
        logger.info(f"2. Statistics: {output_csv}")
        
        logger.info("\nAnalysis Complete")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
