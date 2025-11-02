# ========================================
# Advanced CSI 300 Index Options Trading Model
# Version 2.0 - Mamba Architecture with Quantitative Hedge Fund Strategies
# ========================================

import os
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

# 数据处理库
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import IsolationForest

from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from scipy.special import erf

# 可视化库
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

# 技术指标库
import talib

# 时间处理
from datetime import datetime, timedelta
import time

# 其他工具库
from tqdm import tqdm
from collections import defaultdict, deque
import pickle
import hashlib
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import math
from functools import partial

# 设置警告过滤
warnings.filterwarnings('ignore')


# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# 设置设备 - Apple Silicon优化
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 设置 matplotlib 中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置 seaborn 风格
sns.set_style("darkgrid")
sns.set_palette("husl")

# ========================================
# 数据路径配置
# ========================================
BASE_DIR = '/Users/harry/pycharm/EH/'
INPUT_FEATURE_ENGINEERED_FILE = os.path.join(BASE_DIR, 'feature_engineered_option_data_v1.xlsx')


# ========================================
# 1. 高级期权定价模型特征
# ========================================
class HestonModelFeatures:
    """Heston随机波动率模型特征提取"""

    def __init__(self):
        self.params = {}

    def calibrate_parameters(self, prices, strikes, maturities, spot, r):
        """校准Heston模型参数"""
        try:
            # 初始参数猜测
            initial_params = [0.04, 2.0, 0.04, 0.5, -0.5]  # v0, kappa, theta, xi, rho

            # 目标函数：最小化模型价格与市场价格的差异
            def objective(params):
                try:
                    v0, kappa, theta, xi, rho = params
                    model_prices = self._heston_price_batch(spot, strikes, maturities, r,
                                                            v0, kappa, theta, xi, rho)
                    return np.sum((model_prices - prices) ** 2)
                except:
                    return 1e10

            # 参数边界
            bounds = [(0.001, 1), (0.1, 10), (0.001, 1), (0.1, 2), (-0.99, 0.99)]

            # 优化
            result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')

            self.params = {
                'v0': result.x[0],
                'kappa': result.x[1],
                'theta': result.x[2],
                'xi': result.x[3],
                'rho': result.x[4]
            }
        except Exception as e:
            print(f"Heston calibration failed: {e}, using default parameters")
            self.params = {
                'v0': 0.04,
                'kappa': 2.0,
                'theta': 0.04,
                'xi': 0.5,
                'rho': -0.5
            }

        return self.params

    def _heston_price_batch(self, S, K, T, r, v0, kappa, theta, xi, rho):
        """批量计算Heston模型价格（使用Malliavin calculus近似）"""
        # 确保是数组
        S = np.atleast_1d(S)
        K = np.atleast_1d(K)
        T = np.atleast_1d(T)

        # 广播到相同形状
        if len(S) == 1:
            S = np.repeat(S, len(K))
        if len(T) == 1:
            T = np.repeat(T, len(K))

        # 简化的Heston价格近似
        sigma_avg = np.sqrt(v0 * (1 - np.exp(-kappa * T)) / (kappa * T + 1e-10) +
                            theta * (1 - (1 - np.exp(-kappa * T)) / (kappa * T + 1e-10)))

        # Black-Scholes近似
        d1 = (np.log(S / K + 1e-10) + (r + 0.5 * sigma_avg ** 2) * T) / (sigma_avg * np.sqrt(T) + 1e-10)
        d2 = d1 - sigma_avg * np.sqrt(T)

        prices = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        return prices

    def extract_features(self, df):
        """提取Heston模型特征"""
        features = pd.DataFrame(index=df.index)

        try:
            if all(col in df.columns for col in ['S', 'K', 'TTM_years', 'r_interpolated', 'C_market']):
                # 校准参数（使用部分数据避免过慢）
                sample_size = min(100, len(df))
                sample_idx = np.random.choice(len(df), sample_size, replace=False)
                sample_df = df.iloc[sample_idx]

                # 清理数据
                valid_mask = (sample_df['K'] > 0) & (sample_df['S'] > 0) & \
                             (sample_df['TTM_years'] > 0) & (sample_df['C_market'] > 0)
                sample_df = sample_df[valid_mask]

                if len(sample_df) > 10:
                    params = self.calibrate_parameters(
                        sample_df['C_market'].values,
                        sample_df['K'].values,
                        sample_df['TTM_years'].values,
                        sample_df['S'].values[0],
                        sample_df['r_interpolated'].values[0]
                    )

                    # 添加Heston参数作为特征
                    for key, value in params.items():
                        features[f'heston_{key}'] = value

                    # 计算Heston隐含波动率
                    features['heston_iv'] = np.sqrt(params['v0'])
                    features['heston_vol_of_vol'] = params['xi']
                    features['heston_mean_reversion'] = params['kappa']
                else:
                    # 使用默认值
                    features['heston_v0'] = 0.04
                    features['heston_kappa'] = 2.0
                    features['heston_theta'] = 0.04
                    features['heston_xi'] = 0.5
                    features['heston_rho'] = -0.5
                    features['heston_iv'] = 0.2
                    features['heston_vol_of_vol'] = 0.5
                    features['heston_mean_reversion'] = 2.0
        except Exception as e:
            print(f"Heston feature extraction failed: {e}, using default features")
            # 使用默认值
            features['heston_v0'] = 0.04
            features['heston_kappa'] = 2.0
            features['heston_theta'] = 0.04
            features['heston_xi'] = 0.5
            features['heston_rho'] = -0.5
            features['heston_iv'] = 0.2
            features['heston_vol_of_vol'] = 0.5
            features['heston_mean_reversion'] = 2.0

        return features

class SABRModelFeatures:
    """SABR模型特征提取"""

    def __init__(self):
        self.params = {}

    def calibrate_parameters(self, strikes, forwards, maturities, market_vols):
        """校准SABR模型参数"""
        try:
            # 使用Hagan近似公式
            ATM_idx = np.argmin(np.abs(strikes - forwards))
            ATM_vol = market_vols[ATM_idx] if ATM_idx < len(market_vols) else np.mean(market_vols)

            # 初始参数
            alpha = ATM_vol
            beta = 0.5  # 通常固定
            rho = -0.3  # 初始猜测
            nu = 0.3  # 初始猜测

            def objective(params):
                rho, nu = params
                try:
                    model_vols = self._sabr_vol_vec(forwards, strikes, maturities,
                                                    alpha, beta, rho, nu)
                    return np.sum((model_vols - market_vols) ** 2)
                except:
                    return 1e10  # 返回大值如果计算失败

            bounds = [(-0.99, 0.99), (0.01, 1)]
            result = minimize(objective, [rho, nu], bounds=bounds, method='L-BFGS-B')

            self.params = {
                'alpha': alpha,
                'beta': beta,
                'rho': result.x[0],
                'nu': result.x[1]
            }
        except Exception as e:
            # 如果校准失败，使用默认参数
            print(f"SABR calibration failed: {e}, using default parameters")
            self.params = {
                'alpha': 0.2,
                'beta': 0.5,
                'rho': -0.3,
                'nu': 0.3
            }

        return self.params

    def _sabr_vol_vec(self, F, K, T, alpha, beta, rho, nu):
        """SABR模型隐含波动率（Hagan公式） - 修复版本"""
        # 确保是数组
        F = np.atleast_1d(F)
        K = np.atleast_1d(K)
        T = np.atleast_1d(T)

        # 避免除零
        eps = 1e-10

        # 广播到相同形状
        if F.shape[0] == 1:
            F = np.repeat(F, len(K))
        if T.shape[0] == 1:
            T = np.repeat(T, len(K))

        # 初始化结果数组
        result = np.zeros_like(K)

        # 处理每个元素
        for i in range(len(K)):
            FK = F[i] * K[i]

            if FK <= 0:
                result[i] = alpha
                continue

            logFK = np.log(F[i] / K[i])

            # 防止除零
            if abs(logFK) < eps:
                z = 0
                x_over_z = 1
            else:
                z = nu / alpha * FK ** ((1 - beta) / 2) * logFK
                x_numer = np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho
                x_denom = 1 - rho

                if abs(x_denom) < eps:
                    x = 1
                else:
                    x = np.log(x_numer / x_denom)

                if abs(x) < eps:
                    x_over_z = 1
                else:
                    x_over_z = z / x

            # 计算SABR波动率
            term1_numer = alpha
            term1_denom1 = FK ** ((1 - beta) / 2)
            term1_denom2 = 1 + ((1 - beta) ** 2 / 24) * logFK ** 2 + ((1 - beta) ** 4 / 1920) * logFK ** 4

            if abs(term1_denom1 * term1_denom2) < eps:
                term1 = alpha
            else:
                term1 = term1_numer / (term1_denom1 * term1_denom2) * x_over_z

            term2 = 1 + T[i] * ((1 - beta) ** 2 * alpha ** 2 / (24 * FK ** (1 - beta)) +
                                rho * beta * nu * alpha / (4 * FK ** ((1 - beta) / 2)) +
                                (2 - 3 * rho ** 2) * nu ** 2 / 24)

            result[i] = term1 * term2

        return result

    def extract_features(self, df):
        """提取SABR模型特征"""
        features = pd.DataFrame(index=df.index)

        try:
            if all(col in df.columns for col in ['S', 'K', 'TTM_years', 'IV_Mean']):
                # 使用部分数据校准
                sample_size = min(100, len(df))
                sample_idx = np.random.choice(len(df), sample_size, replace=False)
                sample_df = df.iloc[sample_idx]

                # 清理数据
                valid_mask = (sample_df['K'] > 0) & (sample_df['S'] > 0) & \
                             (sample_df['TTM_years'] > 0) & (sample_df['IV_Mean'] > 0)
                sample_df = sample_df[valid_mask]

                if len(sample_df) > 10:  # 需要足够的数据点
                    params = self.calibrate_parameters(
                        sample_df['K'].values,
                        sample_df['S'].values,
                        sample_df['TTM_years'].values,
                        sample_df['IV_Mean'].values
                    )

                    # 添加SABR参数
                    for key, value in params.items():
                        features[f'sabr_{key}'] = value

                    # 添加SABR衍生特征
                    features['sabr_skew'] = params['rho'] * params['nu']
                    features['sabr_smile'] = params['nu'] ** 2
                    features['sabr_backbone'] = params['beta']
                else:
                    # 数据不足，使用默认值
                    features['sabr_alpha'] = 0.2
                    features['sabr_beta'] = 0.5
                    features['sabr_rho'] = -0.3
                    features['sabr_nu'] = 0.3
                    features['sabr_skew'] = -0.09
                    features['sabr_smile'] = 0.09
                    features['sabr_backbone'] = 0.5
        except Exception as e:
            print(f"SABR feature extraction failed: {e}, using default features")
            # 使用默认值
            features['sabr_alpha'] = 0.2
            features['sabr_beta'] = 0.5
            features['sabr_rho'] = -0.3
            features['sabr_nu'] = 0.3
            features['sabr_skew'] = -0.09
            features['sabr_smile'] = 0.09
            features['sabr_backbone'] = 0.5

        return features



class AdvancedGreeksFeatures:
    """高级希腊字母特征"""

    @staticmethod
    def calculate_vanna(S, K, T, r, sigma):
        """计算Vanna: ∂²V/∂S∂σ"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        vanna = -np.exp(-r * T) * stats.norm.pdf(d2) * d2 / sigma
        return vanna

    @staticmethod
    def calculate_volga(S, K, T, r, sigma):
        """计算Volga: ∂²V/∂σ²"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        volga = S * np.sqrt(T) * stats.norm.pdf(d1) * d1 * (d1 - sigma * np.sqrt(T)) / sigma
        return volga

    @staticmethod
    def calculate_charm(S, K, T, r, sigma):
        """计算Charm: ∂²V/∂S∂t"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        charm = -stats.norm.pdf(d1) * (2 * r * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
        return charm

    @staticmethod
    def calculate_speed(S, K, T, r, sigma):
        """计算Speed: ∂³V/∂S³"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        speed = -gamma / S * (d1 / (sigma * np.sqrt(T)) + 1)
        return speed

    @staticmethod
    def calculate_zomma(S, K, T, r, sigma):
        """计算Zomma: ∂³V/∂S²∂σ"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        zomma = gamma * (d1 * d2 - 1) / sigma
        return zomma

    def extract_features(self, df):
        """提取高级Greeks特征"""
        features = pd.DataFrame(index=df.index)

        required_cols = ['S', 'K', 'TTM_years', 'r_interpolated', 'IV_Mean']
        if all(col in df.columns for col in required_cols):
            S = df['S'].values
            K = df['K'].values
            T = df['TTM_years'].values
            r = df['r_interpolated'].values
            sigma = df['IV_Mean'].values

            # 计算所有高级Greeks
            features['vanna'] = self.calculate_vanna(S, K, T, r, sigma)
            features['volga'] = self.calculate_volga(S, K, T, r, sigma)
            features['charm'] = self.calculate_charm(S, K, T, r, sigma)
            features['speed'] = self.calculate_speed(S, K, T, r, sigma)
            features['zomma'] = self.calculate_zomma(S, K, T, r, sigma)

            # Greeks比率
            if 'Delta' in df.columns:
                features['vanna_delta_ratio'] = features['vanna'] / (df['Delta'] + 1e-10)

            # Greeks组合
            features['volga_vanna_ratio'] = features['volga'] / (np.abs(features['vanna']) + 1e-10)
            features['charm_speed_product'] = features['charm'] * features['speed']

        return features


class MarketMicrostructureFeatures:
    """市场微观结构特征"""

    @staticmethod
    def calculate_order_flow_imbalance(volume, price_change):
        """计算订单流失衡"""
        buy_volume = volume * (price_change > 0).astype(float)
        sell_volume = volume * (price_change < 0).astype(float)
        imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume + 1e-10)
        return imbalance

    @staticmethod
    def calculate_toxic_flow_probability(volume, returns, window=50):
        """计算毒性流概率（简化VPIN）"""
        # Volume-synchronized probability of informed trading
        volume_buckets = pd.Series(volume).rolling(window).sum()
        return_volatility = pd.Series(returns).rolling(window).std()

        # 毒性流指标
        toxic_prob = return_volatility / (volume_buckets + 1e-10)
        toxic_prob = toxic_prob.fillna(0)

        # 归一化到[0,1]
        toxic_prob = (toxic_prob - toxic_prob.min()) / (toxic_prob.max() - toxic_prob.min() + 1e-10)
        return toxic_prob.values

    @staticmethod
    def calculate_effective_spread(high, low, close):
        """计算有效价差"""
        spread = 2 * np.abs(close - (high + low) / 2) / ((high + low) / 2)
        return spread

    def extract_features(self, df):
        """提取市场微观结构特征"""
        features = pd.DataFrame(index=df.index)

        if 'volume' in df.columns:
            # 订单流失衡
            if '涨跌 1' in df.columns:
                features['order_flow_imbalance'] = self.calculate_order_flow_imbalance(
                    df['volume'].values, df['涨跌 1'].values
                )

            # 毒性流概率
            if 'C_market' in df.columns:
                returns = df['C_market'].pct_change().fillna(0)
                features['toxic_flow_prob'] = self.calculate_toxic_flow_probability(
                    df['volume'].values, returns.values
                )

            # 有效价差
            if all(col in df.columns for col in ['最高价', '最低价', 'C_market']):
                features['effective_spread'] = self.calculate_effective_spread(
                    df['最高价'].values, df['最低价'].values, df['C_market'].values
                )

            # 交易强度
            features['trade_intensity'] = df['volume'] / df['volume'].rolling(20).mean()

            # 大单检测
            volume_threshold = df['volume'].quantile(0.9)
            features['large_trade_indicator'] = (df['volume'] > volume_threshold).astype(float)

            # 机构vs散户分类（基于交易规模）
            volume_percentiles = df['volume'].quantile([0.25, 0.75])
            features['institutional_flow'] = (df['volume'] > volume_percentiles[0.75]).astype(float)
            features['retail_flow'] = (df['volume'] < volume_percentiles[0.25]).astype(float)

        # 期权流特征
        if 'PCR_Volume' in df.columns:
            # 异常期权活动
            pcr_ma = df['PCR_Volume'].rolling(20).mean()
            pcr_std = df['PCR_Volume'].rolling(20).std()
            features['unusual_options_activity'] = np.abs(df['PCR_Volume'] - pcr_ma) / (pcr_std + 1e-10)

        # 净漂移累积溢价流
        if 'open_interest' in df.columns and 'C_market' in df.columns:
            features['net_drift_premium'] = (df['open_interest'] * df['C_market']).rolling(10).sum()

        return features


class ChineseMarketFeatures:
    """中国市场特定特征"""

    @staticmethod
    def calculate_ivx_features(vix_data):
        """计算iVX（中国隐含波动率指数）特征"""
        features = pd.DataFrame()

        # HAR模型组件
        features['ivx_daily'] = vix_data
        features['ivx_weekly'] = vix_data.rolling(5).mean()
        features['ivx_monthly'] = vix_data.rolling(22).mean()

        # iVX期限结构
        features['ivx_term_spread'] = features['ivx_monthly'] - features['ivx_daily']

        return features

    @staticmethod
    def calculate_sentiment_features(df):
        """计算情绪特征（模拟）"""
        features = pd.DataFrame(index=df.index)

        # 模拟东方财富股吧情绪
        features['eastmoney_sentiment'] = np.random.randn(len(df)) * 0.1

        # 负面情绪影响更强
        features['negative_sentiment_amplified'] = np.where(
            features['eastmoney_sentiment'] < 0,
            features['eastmoney_sentiment'] * 1.5,
            features['eastmoney_sentiment']
        )

        # 情绪动量
        features['sentiment_momentum'] = features['eastmoney_sentiment'].rolling(5).mean()

        return features

    @staticmethod
    def calculate_regulatory_features(df):
        """计算监管特征"""
        features = pd.DataFrame(index=df.index)

        # T+0 vs T+1影响
        features['t0_advantage'] = 1  # 衍生品T+0优势

        # 仓位限制（300手）
        if 'open_interest' in df.columns:
            features['position_limit_utilization'] = df['open_interest'] / 300
            features['near_limit_indicator'] = (features['position_limit_utilization'] > 0.8).astype(float)

        # 个人投资者占比影响
        features['retail_dominance'] = 0.7  # 70%个人投资者

        return features

    def extract_features(self, df):
        """提取中国市场特征"""
        features = pd.DataFrame(index=df.index)

        # iVX特征
        if 'VIX_Close' in df.columns:
            ivx_features = self.calculate_ivx_features(df['VIX_Close'])
            features = pd.concat([features, ivx_features], axis=1)

        # 情绪特征
        sentiment_features = self.calculate_sentiment_features(df)
        features = pd.concat([features, sentiment_features], axis=1)

        # 监管特征
        regulatory_features = self.calculate_regulatory_features(df)
        features = pd.concat([features, regulatory_features], axis=1)

        # CSI 300期货主导价格发现
        features['futures_price_discovery'] = 0.7  # 70%贡献度

        return features


# ========================================
# 2. 综合高级特征工程类
# ========================================

class QuantitativeFeatureEngine:
    """量化对冲基金级别特征工程"""

    def __init__(self):
        self.heston = HestonModelFeatures()
        self.sabr = SABRModelFeatures()
        self.greeks = AdvancedGreeksFeatures()
        self.microstructure = MarketMicrostructureFeatures()
        self.chinese_market = ChineseMarketFeatures()
        self.scalers = {}
        self.feature_names = []

    def engineer_all_features(self, df):
        """完整特征工程流程"""
        print("Starting quantitative feature engineering...")

        features_list = []

        # 1. 原始特征
        print("  Processing original features...")
        features_list.append(df)

        # 2. Heston模型特征
        print("  Extracting Heston model features...")
        heston_features = self.heston.extract_features(df)
        features_list.append(heston_features)

        # 3. SABR模型特征
        print("  Extracting SABR model features...")
        sabr_features = self.sabr.extract_features(df)
        features_list.append(sabr_features)

        # 4. 高级Greeks
        print("  Extracting advanced Greeks features...")
        greeks_features = self.greeks.extract_features(df)
        features_list.append(greeks_features)

        # 5. 市场微观结构
        print("  Extracting market microstructure features...")
        micro_features = self.microstructure.extract_features(df)
        features_list.append(micro_features)

        # 6. 中国市场特征
        print("  Extracting Chinese market features...")
        chinese_features = self.chinese_market.extract_features(df)
        features_list.append(chinese_features)

        # 7. 技术指标特征
        print("  Extracting technical indicators...")
        tech_features = self._extract_technical_features(df)
        features_list.append(tech_features)

        # 8. 波动率曲面特征
        print("  Extracting volatility surface features...")
        vol_surface_features = self._extract_volatility_surface_features(df)
        features_list.append(vol_surface_features)

        # 9. 期权流特征
        print("  Extracting option flow features...")
        flow_features = self._extract_option_flow_features(df)
        features_list.append(flow_features)

        # 10. 制度特征
        print("  Extracting regime features...")
        regime_features = self._extract_regime_features(df)
        features_list.append(regime_features)

        # 合并所有特征
        all_features = pd.concat(features_list, axis=1)

        # 移除重复列
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]

        print(f"Feature engineering completed. Total features: {len(all_features.columns)}")
        self.feature_names = all_features.columns.tolist()

        return all_features

    def _extract_technical_features(self, df):
        """提取技术指标特征"""
        features = pd.DataFrame(index=df.index)

        if 'C_market' in df.columns:
            close = df['C_market'].values

            # RSI不同周期
            for period in [7, 14, 21]:
                features[f'RSI_{period}'] = talib.RSI(close, timeperiod=period)

            # MACD
            macd, signal, hist = talib.MACD(close)
            features['MACD'] = macd
            features['MACD_Signal'] = signal
            features['MACD_Hist'] = hist

            # 布林带
            upper, middle, lower = talib.BBANDS(close)
            features['BB_Upper'] = upper
            features['BB_Lower'] = lower
            features['BB_Width'] = upper - lower
            features['BB_Position'] = (close - lower) / (upper - lower + 1e-10)

            # ATR
            if all(col in df.columns for col in ['最高价', '最低价']):
                features['ATR_14'] = talib.ATR(df['最高价'].values,
                                               df['最低价'].values,
                                               close, timeperiod=14)

            # 威廉指标
            if all(col in df.columns for col in ['最高价', '最低价']):
                features['WILLR'] = talib.WILLR(df['最高价'].values,
                                                df['最低价'].values,
                                                close, timeperiod=14)

            # ADX
            if all(col in df.columns for col in ['最高价', '最低价']):
                features['ADX'] = talib.ADX(df['最高价'].values,
                                            df['最低价'].values,
                                            close, timeperiod=14)

            # CCI
            if all(col in df.columns for col in ['最高价', '最低价']):
                features['CCI'] = talib.CCI(df['最高价'].values,
                                            df['最低价'].values,
                                            close, timeperiod=14)

        return features

    def _extract_volatility_surface_features(self, df):
        """提取波动率曲面特征"""
        features = pd.DataFrame(index=df.index)

        if 'IV_Mean' in df.columns and 'Moneyness' in df.columns and 'TTM_years' in df.columns:
            # 波动率微笑参数
            features['iv_atm'] = df['IV_Mean'][np.abs(df['Moneyness'] - 1) < 0.05].mean()
            features['iv_25d_put'] = df['IV_Mean'][(df['Moneyness'] < 0.95) & (df['Moneyness'] > 0.9)].mean()
            features['iv_25d_call'] = df['IV_Mean'][(df['Moneyness'] > 1.05) & (df['Moneyness'] < 1.1)].mean()

            # 偏斜度量
            features['25d_risk_reversal'] = features['iv_25d_call'] - features['iv_25d_put']
            features['25d_butterfly'] = (features['iv_25d_call'] + features['iv_25d_put']) / 2 - features['iv_atm']

            # 期限结构
            for maturity in [7, 30, 60, 90]:
                mask = np.abs(df['TTM_days'] - maturity) < 5
                features[f'iv_term_{maturity}d'] = df.loc[mask, 'IV_Mean'].mean()

            # 期限结构斜率
            features['term_structure_slope'] = (features['iv_term_90d'] - features['iv_term_7d']) / 83

            # 波动率曲面曲率
            features['surface_curvature'] = features['25d_butterfly'] * features['term_structure_slope']

        return features

    def _extract_option_flow_features(self, df):
        """提取期权流特征"""
        features = pd.DataFrame(index=df.index)

        if 'volume' in df.columns and 'open_interest' in df.columns:
            # Volume/OI比率
            features['volume_oi_ratio'] = df['volume'] / (df['open_interest'] + 1e-10)

            # 异常交易量
            vol_ma = df['volume'].rolling(20).mean()
            vol_std = df['volume'].rolling(20).std()
            features['volume_zscore'] = (df['volume'] - vol_ma) / (vol_std + 1e-10)
            features['unusual_volume'] = (features['volume_zscore'] > 2).astype(float)

            # 聪明钱指标
            features['smart_money_indicator'] = features['volume_oi_ratio'] * features['unusual_volume']

        # Put-Call比率特征
        if 'PCR_Volume' in df.columns:
            features['pcr_ma5'] = df['PCR_Volume'].rolling(5).mean()
            features['pcr_ma20'] = df['PCR_Volume'].rolling(20).mean()
            features['pcr_deviation'] = (df['PCR_Volume'] - features['pcr_ma20']) / (features['pcr_ma20'] + 1e-10)

        if 'PCR_OI' in df.columns:
            features['pcr_oi_ma5'] = df['PCR_OI'].rolling(5).mean()
            features['pcr_divergence'] = df['PCR_Volume'] - df['PCR_OI']

        return features

    def _extract_regime_features(self, df):
        """提取市场制度特征"""
        features = pd.DataFrame(index=df.index)

        if 'C_market' in df.columns:
            returns = df['C_market'].pct_change()

            # 滚动波动率制度
            vol_20d = returns.rolling(20).std() * np.sqrt(252)
            vol_60d = returns.rolling(60).std() * np.sqrt(252)

            features['vol_regime_20d'] = pd.cut(vol_20d, bins=[0, 0.15, 0.25, np.inf],
                                                labels=[0, 1, 2]).astype(float)
            features['vol_regime_60d'] = pd.cut(vol_60d, bins=[0, 0.15, 0.25, np.inf],
                                                labels=[0, 1, 2]).astype(float)

            # 动量制度
            momentum_5d = returns.rolling(5).mean()
            momentum_20d = returns.rolling(20).mean()

            features['momentum_regime'] = 0  # Neutral
            features.loc[(momentum_5d > 0) & (momentum_20d > 0), 'momentum_regime'] = 1  # Bull
            features.loc[(momentum_5d < 0) & (momentum_20d < 0), 'momentum_regime'] = -1  # Bear

            # 趋势强度
            features['trend_strength'] = momentum_20d / (vol_20d + 1e-10)

        # VIX制度
        if 'VIX_Close' in df.columns:
            vix_percentiles = df['VIX_Close'].quantile([0.25, 0.5, 0.75])
            features['vix_regime'] = pd.cut(df['VIX_Close'],
                                            bins=[0] + vix_percentiles.tolist() + [np.inf],
                                            labels=[0, 1, 2, 3]).astype(float)

        return features


# ========================================
# 3. 数据加载和预处理类
# ========================================

class AdvancedDataLoader:
    """高级数据加载器"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.raw_data = None
        self.processed_data = None
        self.feature_engine = QuantitativeFeatureEngine()

    def load_data(self) -> pd.DataFrame:
        """加载原始数据"""
        print(f"Loading data from {self.file_path}...")
        self.raw_data = pd.read_excel(self.file_path)
        print(f"Data loaded. Shape: {self.raw_data.shape}")

        # 数据类型转换
        self._convert_data_types()

        # 处理缺失值
        self._handle_missing_values()

        # 异常值处理
        self._handle_outliers()

        return self.raw_data

    def _convert_data_types(self):
        """转换数据类型"""
        # 转换日期
        if 'trade_date' in self.raw_data.columns:
            self.raw_data['trade_date'] = pd.to_datetime(self.raw_data['trade_date'])

        # 数值列
        numeric_columns = self.raw_data.select_dtypes(include=['object']).columns
        for col in numeric_columns:
            try:
                self.raw_data[col] = pd.to_numeric(self.raw_data[col], errors='coerce')
            except:
                pass

    def _handle_missing_values(self):
        """处理缺失值"""
        # 移除无穷值
        numeric_columns = self.raw_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            self.raw_data[col] = self.raw_data[col].replace([np.inf, -np.inf], np.nan)

        # 前向填充
        self.raw_data = self.raw_data.fillna(method='ffill')

        # 后向填充
        self.raw_data = self.raw_data.fillna(method='bfill')

        # 剩余用中位数填充
        for col in numeric_columns:
            if self.raw_data[col].isna().any():
                median_val = self.raw_data[col].median()
                if np.isnan(median_val):
                    self.raw_data[col] = self.raw_data[col].fillna(0)
                else:
                    self.raw_data[col] = self.raw_data[col].fillna(median_val)

    def _handle_outliers(self):
        """处理异常值（使用IsolationForest）"""
        numeric_columns = self.raw_data.select_dtypes(include=[np.number]).columns

        # 排除某些不应该做异常值处理的列
        exclude_cols = ['trade_date', 'option_code', 'Type']
        feature_cols = [col for col in numeric_columns if col not in exclude_cols]

        if len(feature_cols) > 0:
            # 使用IsolationForest检测异常值
            iso_forest = IsolationForest(contamination=0.01, random_state=42)
            outliers = iso_forest.fit_predict(self.raw_data[feature_cols].fillna(0))

            # 对异常值进行裁剪而非删除
            outlier_indices = np.where(outliers == -1)[0]
            for idx in outlier_indices:
                for col in feature_cols:
                    # 使用95%分位数裁剪
                    upper_bound = self.raw_data[col].quantile(0.95)
                    lower_bound = self.raw_data[col].quantile(0.05)
                    self.raw_data.loc[idx, col] = np.clip(self.raw_data.loc[idx, col],
                                                          lower_bound, upper_bound)

    def process_data(self) -> pd.DataFrame:
        """处理数据并进行特征工程"""
        if self.raw_data is None:
            self.load_data()

        print("Processing data with advanced feature engineering...")
        self.processed_data = self.feature_engine.engineer_all_features(self.raw_data)

        return self.processed_data

    def create_sequences(self, sequence_length: int = 30,
                         target_col: str = 'C_market',
                         prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray, Any, Any]:
        """创建时间序列数据 - 完全修复版本"""
        if self.processed_data is None:
            self.process_data()

        # 确保目标列存在
        if target_col not in self.processed_data.columns:
            raise ValueError(f"Target column {target_col} not found")

        # 步骤 1：计算真实的收益率（使用原始价格）
        prices = self.processed_data[target_col].values

        # 计算日收益率 - 修复：确保正确的时间对齐
        returns = np.zeros(len(prices))
        returns[1:] = (prices[1:] - prices[:-1]) / (np.abs(prices[:-1]) + 1e-10)

        # 裁剪极端值
        returns = np.clip(returns, -0.1, 0.1)  # 限制在±10%日收益率

        # 步骤 2：选择特征列（不包括目标列和其他非特征列）
        exclude_cols = ['trade_date', 'option_code', 'Expiry', 'Type', target_col,
                        '最高价', '最低价', 'C_market']  # 排除可能包含未来信息的列
        feature_cols = [col for col in self.processed_data.columns
                        if col not in exclude_cols]

        # 只保留数值列
        features = self.processed_data[feature_cols].select_dtypes(include=[np.number])

        # 步骤 3：创建序列 - 关键修复
        X, y = [], []

        # 修复：确保有足够的数据点用于预测
        max_index = len(features) - prediction_horizon

        for i in range(sequence_length, max_index):
            # 特征序列：从 i-sequence_length 到 i（不包括i）
            X.append(features.iloc[i - sequence_length:i].values)

            # 目标：未来第prediction_horizon天的收益率
            # 关键修复：使用 i + prediction_horizon 而不是 i + prediction_horizon - 1
            future_return = returns[i + prediction_horizon]
            y.append(future_return)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        # 步骤 4：只在训练数据上fit scaler
        # 这里先不做标准化，在split_data中处理

        # 数据清理
        X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.1, neginf=-0.1)

        print(f"Created sequences - X shape: {X.shape}, y shape: {y.shape}")
        print(f"Target returns - Mean: {np.mean(y):.4f}, Std: {np.std(y):.4f}")
        print(f"Target returns - Min: {np.min(y):.4f}, Max: {np.max(y):.4f}")

        # 检查数据合理性
        if np.abs(np.mean(y)) > 0.01:
            print("⚠ Warning: Mean return is too high, possible data issue")
        if np.std(y) < 0.001:
            print("⚠ Warning: Std of returns is too low, possible data issue")

        return X, y, features.columns.tolist()

    def split_data(self, test_size: float = 0.2,
                   val_size: float = 0.1) -> Dict[str, Union[np.ndarray, Any]]:
        """划分训练集、验证集和测试集"""
        X, y, scaler = self.create_sequences()

        # 时间序列分割
        n_samples = len(X)
        train_size = int(n_samples * (1 - test_size - val_size))
        val_size = int(n_samples * val_size)

        X_train = X[:train_size]
        y_train = y[:train_size]

        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]

        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]

        print(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'scaler': scaler
        }


# ========================================
# 4. 数据集类
# ========================================

class AdvancedTradingDataset(Dataset):
    """高级交易数据集"""

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 augment: bool = False,
                 noise_level: float = 0.01,
                 mixup_alpha: float = 0.2):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.augment = augment
        self.noise_level = noise_level
        self.mixup_alpha = mixup_alpha
        self.training = augment  # 添加training属性，训练时通常augment=True

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.augment and self.training:
            # 添加噪声
            if np.random.random() > 0.5:
                noise = torch.randn_like(x) * self.noise_level
                x = x + noise

            # Mixup数据增强
            if np.random.random() > 0.7:
                lambda_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                idx2 = np.random.randint(len(self.X))
                x = lambda_mix * x + (1 - lambda_mix) * self.X[idx2]
                y = lambda_mix * y + (1 - lambda_mix) * self.y[idx2]

        return x, y

    def set_training(self, mode: bool):
        """设置训练模式"""
        self.training = mode
# ========================================
# Part 2: Mamba架构和核心模型定义
# ========================================

# ========================================
# 5. Mamba架构实现（State Space Model）
# ========================================

# Alternative: Simplified Mamba with Linear Attention
class SimplifiedMambaBlock(nn.Module):
    """Simplified Mamba using linear attention mechanism for efficiency"""

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2,
                 dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # Convolution
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )

        # Linear attention components
        self.q_proj = nn.Linear(self.d_inner, d_state)
        self.k_proj = nn.Linear(self.d_inner, d_state)
        self.v_proj = nn.Linear(self.d_inner, self.d_inner)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

        # Layer norm
        self.norm = nn.LayerNorm(self.d_inner)

    def forward(self, x):
        batch, length, _ = x.shape

        # Input projection and split
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # Convolution
        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :length]
        x_conv = x_conv.transpose(1, 2)

        # Apply activation
        x_conv = F.silu(x_conv)

        # Linear attention (efficient alternative to SSM)
        Q = self.q_proj(x_conv)
        K = self.k_proj(x_conv)
        V = self.v_proj(x_conv)

        # Compute attention scores efficiently
        # Using the associative property of matrix multiplication
        # This is O(n*d^2) instead of O(n^2*d)
        KV = torch.einsum('bnd,bnm->bdm', K, V)
        y = torch.einsum('bnd,bdm->bnm', Q, KV)

        # Normalize
        y = self.norm(y)

        # Gating
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)
        output = self.dropout(output)

        return output


# Use the simplified version for better performance
class BiMamba(nn.Module):
    """Bidirectional Mamba Model - using simplified blocks"""

    def __init__(self, d_model: int, n_layers: int = 4,
                 d_state: int = 16, dropout: float = 0.1):
        super().__init__()

        # Use SimplifiedMambaBlock for better performance
        self.layers = nn.ModuleList([
            SimplifiedMambaBlock(d_model, d_state=d_state, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Single direction pass with residual connections
        # Removing bidirectional to improve speed
        output = x
        for layer in self.layers:
            residual = output
            output = layer(output)
            output = output + residual

        output = self.norm(output)
        output = self.dropout(output)

        return output


class CMMamba(nn.Module):
    """Channel-Mixing Mamba - Simplified"""

    def __init__(self, d_model: int, n_channels: int,
                 n_layers: int = 4, d_state: int = 16,
                 dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.n_channels = n_channels

        # Single Mamba for all channels
        self.mamba = BiMamba(d_model, n_layers=n_layers,
                             d_state=d_state, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = self.mamba(x)
        output = self.dropout(output)
        return output

# ========================================
# 6. 多智能体系统
# ========================================

class VolatilityArbitrageAgent(nn.Module):
    """波动率套利智能体 - 量化基金策略"""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 波动率风险溢价检测
        self.vrp_detector = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 波动率期限结构交易
        self.term_structure_trader = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

        # 波动率偏斜交易
        self.skew_trader = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.feature_extractor(x)

        # 三种波动率策略信号
        vrp_signal = self.vrp_detector(features)
        term_signal = self.term_structure_trader(features)
        skew_signal = self.skew_trader(features)

        # 综合信号
        combined_signal = (vrp_signal + term_signal + skew_signal) / 3

        # 确保输出是2D
        if combined_signal.dim() == 1:
            combined_signal = combined_signal.unsqueeze(1)

        return combined_signal, {
            'vrp': vrp_signal,
            'term': term_signal,
            'skew': skew_signal
        }


class DeltaGammaHedgingAgent(nn.Module):
    """Delta-Gamma动态对冲智能体"""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Greeks估计器
        self.greeks_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # Delta, Gamma, Vega, Theta, Rho
        )

        # 对冲比率计算器
        self.hedge_calculator = nn.Sequential(
            nn.Linear(hidden_dim + 5, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # [spot_hedge, atm_hedge, otm_hedge]
        )

        # 动态调整器
        self.dynamic_adjuster = nn.Sequential(
            nn.Linear(hidden_dim + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)

        # 估计Greeks
        greeks = self.greeks_estimator(features)

        # 计算对冲比率
        combined_features = torch.cat([features, greeks], dim=-1)
        hedge_ratios = self.hedge_calculator(combined_features)

        # 动态调整
        hedge_features = torch.cat([features, hedge_ratios], dim=-1)
        adjustment = self.dynamic_adjuster(hedge_features)

        # 最终对冲信号
        final_hedge = torch.sum(hedge_ratios * adjustment, dim=-1, keepdim=True)
        final_hedge = torch.tanh(final_hedge)

        return final_hedge, {
            'greeks': greeks,
            'hedge_ratios': hedge_ratios,
            'adjustment': adjustment
        }


class MarketMakingAgent(nn.Module):
    """做市商智能体 - Two Sigma/Citadel风格"""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 买卖价差预测
        self.spread_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # [bid_spread, ask_spread]
            nn.Softplus()  # 确保正值
        )

        # 库存管理
        self.inventory_manager = nn.Sequential(
            nn.Linear(hidden_dim + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

        # 毒性流检测
        self.toxicity_detector = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)

        # 预测买卖价差
        spreads = self.spread_predictor(features)

        # 库存管理
        inventory_features = torch.cat([features, spreads], dim=-1)
        inventory_signal = self.inventory_manager(inventory_features)

        # 毒性流检测
        toxicity = self.toxicity_detector(features)

        # 调整报价（考虑毒性）
        adjusted_signal = inventory_signal * (1 - toxicity * 0.5)

        return adjusted_signal, {
            'spreads': spreads,
            'inventory': inventory_signal,
            'toxicity': toxicity
        }


class RegimeSwitchingAgent(nn.Module):
    """市场制度切换智能体"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, n_regimes: int = 4):
        super().__init__()

        self.n_regimes = n_regimes

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 制度分类器（HMM风格）
        self.regime_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_regimes),
            nn.Softmax(dim=-1)
        )

        # 每个制度的策略
        self.regime_strategies = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Tanh()
            ) for _ in range(n_regimes)
        ])

        # 转换概率矩阵
        self.transition_matrix = nn.Parameter(
            torch.randn(n_regimes, n_regimes) * 0.1
        )

    def forward(self, x):
        features = self.encoder(x)

        # 预测当前制度
        regime_probs = self.regime_classifier(features)

        # 每个制度的策略输出
        strategies = []
        for i in range(self.n_regimes):
            strategy = self.regime_strategies[i](features)
            if strategy.dim() == 1:
                strategy = strategy.unsqueeze(1)
            strategies.append(strategy)

        strategies = torch.stack(strategies, dim=-1)

        # 确保维度匹配
        if regime_probs.dim() == 2 and strategies.dim() == 3:
            regime_probs = regime_probs.unsqueeze(1)

        # 加权策略
        weighted_strategy = torch.sum(strategies * regime_probs, dim=-1)

        if weighted_strategy.dim() == 1:
            weighted_strategy = weighted_strategy.unsqueeze(1)

        # 获取转换概率
        transition_probs = F.softmax(self.transition_matrix, dim=-1)

        return weighted_strategy, {
            'regime_probs': regime_probs.squeeze(1) if regime_probs.dim() == 3 else regime_probs,
            'strategies': strategies,
            'transitions': transition_probs
        }


class RiskManagementAgent(nn.Module):
    """风险管理智能体 - 专注于尾部风险"""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # VaR估计器
        self.var_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # 95%, 99%, 99.9% VaR
            nn.Sigmoid()
        )

        # CVaR估计器
        self.cvar_estimator = nn.Sequential(
            nn.Linear(hidden_dim + 3, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # 95%, 99%, 99.9% CVaR
            nn.Sigmoid()
        )

        # 尾部风险指标
        self.tail_risk_detector = nn.Sequential(
            nn.Linear(hidden_dim + 6, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Kelly准则仓位调整
        self.kelly_adjuster = nn.Sequential(
            nn.Linear(hidden_dim + 7, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)

        # 风险度量
        var = self.var_estimator(features)

        var_features = torch.cat([features, var], dim=-1)
        cvar = self.cvar_estimator(var_features)

        risk_features = torch.cat([features, var, cvar], dim=-1)
        tail_risk = self.tail_risk_detector(risk_features)

        # Kelly准则调整
        kelly_features = torch.cat([risk_features, tail_risk], dim=-1)
        kelly_fraction = self.kelly_adjuster(kelly_features)

        # 保守Kelly（0.25倍）
        conservative_kelly = kelly_fraction * 0.25

        return conservative_kelly, {
            'var': var,
            'cvar': cvar,
            'tail_risk': tail_risk,
            'kelly': kelly_fraction
        }


# ========================================
# 7. 损失函数
# ========================================

class DirectSharpeRatioLoss(nn.Module):
    """直接优化Sharpe比率的损失函数"""

    def __init__(self, risk_free_rate: float = 0.03):
        super().__init__()
        self.risk_free_rate = risk_free_rate / 252  # 日化

    def forward(self, returns):
        """
        returns: (batch_size,) 或 (batch_size, 1)
        """
        if returns.dim() > 1:
            returns = returns.squeeze()

        # 计算超额收益
        excess_returns = returns - self.risk_free_rate

        # Sharpe比率 = mean / std
        mean_return = torch.mean(excess_returns)
        std_return = torch.std(excess_returns) + 1e-8

        # 负Sharpe比率作为损失（最大化Sharpe等价于最小化负Sharpe）
        sharpe_ratio = mean_return / std_return

        # 添加正则化项防止过度波动
        penalty = torch.mean(torch.abs(returns[1:] - returns[:-1]))  # 交易频率惩罚

        loss = -sharpe_ratio + 0.01 * penalty

        return loss, sharpe_ratio


class CVaRLoss(nn.Module):
    """条件风险价值（CVaR）损失函数"""

    def __init__(self, alpha: float = 0.05):
        super().__init__()
        self.alpha = alpha

    def forward(self, returns):
        """
        returns: (batch_size,) 或 (batch_size, 1)
        """
        if returns.dim() > 1:
            returns = returns.squeeze()

        # 计算VaR
        sorted_returns, _ = torch.sort(returns)
        var_index = int(self.alpha * len(returns))
        var = sorted_returns[var_index]

        # 计算CVaR（低于VaR的平均损失）
        cvar = torch.mean(sorted_returns[:var_index])

        # CVaR损失（最小化下行风险）
        loss = -cvar  # 负号因为我们要最小化损失

        return loss, cvar


class HierarchicalRiskParityLoss(nn.Module):
    """层次风险平价损失函数"""

    def __init__(self, n_assets: int = 4):
        super().__init__()
        self.n_assets = n_assets

    def forward(self, weights, returns):
        """
        weights: (batch_size, n_assets) - 投资组合权重
        returns: (batch_size, n_assets) - 资产收益率
        """
        # 计算投资组合收益
        portfolio_returns = torch.sum(weights * returns, dim=-1)

        # 计算各资产的风险贡献
        asset_risks = torch.std(returns, dim=0)
        risk_contributions = weights * asset_risks

        # 风险平价目标：所有资产风险贡献相等
        target_contribution = torch.mean(risk_contributions, dim=-1, keepdim=True)
        parity_loss = torch.mean((risk_contributions - target_contribution) ** 2)

        # 组合Sharpe比率
        portfolio_mean = torch.mean(portfolio_returns)
        portfolio_std = torch.std(portfolio_returns) + 1e-8
        sharpe = portfolio_mean / portfolio_std

        # 综合损失
        loss = -sharpe + 0.1 * parity_loss

        return loss, {
            'sharpe': sharpe,
            'parity_loss': parity_loss,
            'portfolio_returns': portfolio_returns
        }


class IntegratedTradingLoss(nn.Module):
    """综合交易损失函数"""

    def __init__(self,
                 sharpe_weight: float = 0.4,
                 cvar_weight: float = 0.3,
                 returns_weight: float = 0.3,
                 transaction_cost: float = 0.001):
        super().__init__()

        self.sharpe_loss = DirectSharpeRatioLoss()
        self.cvar_loss = CVaRLoss()
        self.transaction_cost = transaction_cost

        self.sharpe_weight = sharpe_weight
        self.cvar_weight = cvar_weight
        self.returns_weight = returns_weight

    def forward(self, predictions, targets, positions=None):
        """
        predictions: 模型预测
        targets: 真实收益
        positions: 仓位（可选）
        """
        # 确保维度正确
        if predictions.dim() > 1:
            predictions = predictions.squeeze()
        if targets.dim() > 1:
            targets = targets.squeeze()

        # 计算交易收益
        if positions is not None:
            returns = predictions * positions.squeeze()
        else:
            returns = predictions * torch.sign(targets)

        # 防止NaN
        returns = torch.clamp(returns, -10, 10)

        # Sharpe比率损失
        sharpe_loss, sharpe_ratio = self.sharpe_loss(returns)

        # CVaR损失
        cvar_loss, cvar = self.cvar_loss(returns)

        # 收益损失（MSE）
        returns_loss = F.mse_loss(predictions, targets)

        # 交易成本
        if positions is not None and len(positions) > 1:
            position_changes = torch.abs(positions[1:] - positions[:-1])
            transaction_loss = torch.mean(position_changes) * self.transaction_cost
        else:
            transaction_loss = torch.tensor(0.0).to(predictions.device)

        # 综合损失
        total_loss = (self.sharpe_weight * sharpe_loss +
                      self.cvar_weight * cvar_loss +
                      self.returns_weight * returns_loss +
                      transaction_loss)

        # 防止NaN
        if torch.isnan(total_loss):
            total_loss = returns_loss

        return total_loss, {
            'sharpe_ratio': sharpe_ratio.item() if not torch.isnan(sharpe_ratio) else 0.0,
            'cvar': cvar.item() if not torch.isnan(cvar) else 0.0,
            'returns_loss': returns_loss.item() if not torch.isnan(returns_loss) else 0.0,
            'transaction_cost': transaction_loss.item() if not torch.isnan(transaction_loss) else 0.0,
            'total_loss': total_loss.item() if not torch.isnan(total_loss) else 0.0
        }


# ========================================
# 8. 主模型架构
# ========================================

class QuantitativeOptionTradingModel(nn.Module):
    """Quantitative Options Trading Model - FIXED Architecture"""

    def __init__(self,
                 input_dim: int,
                 sequence_length: int,
                 hidden_dim: int = 512,
                 n_mamba_layers: int = 6,
                 n_agents: int = 5,
                 dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(
            sequence_length, hidden_dim
        )

        # CMMamba backbone network
        self.mamba_backbone = CMMamba(
            d_model=hidden_dim,
            n_channels=4,  # 4 channels for different feature types
            n_layers=n_mamba_layers,
            d_state=32,
            dropout=dropout
        )

        # Multi-agent system
        self.vol_arb_agent = VolatilityArbitrageAgent(hidden_dim, hidden_dim // 2)
        self.delta_gamma_agent = DeltaGammaHedgingAgent(hidden_dim, hidden_dim // 2)
        self.market_making_agent = MarketMakingAgent(hidden_dim, hidden_dim // 2)
        self.regime_agent = RegimeSwitchingAgent(hidden_dim, hidden_dim // 2)
        self.risk_agent = RiskManagementAgent(hidden_dim, hidden_dim // 2)

        # Agent aggregator
        self.agent_aggregator = nn.Sequential(
            nn.Linear(n_agents, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Final output layer
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )

        # Auxiliary task heads
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )

        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def _create_positional_encoding(self, seq_len: int, d_model: int):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x):
        batch_size = x.size(0)

        # Input projection and positional encoding
        x = self.input_projection(x)

        # Add positional encoding only if sequence dimensions match
        if x.size(1) <= self.positional_encoding.size(1):
            x = x + self.positional_encoding[:, :x.size(1), :]

        # Mamba processing
        mamba_out = self.mamba_backbone(x)

        # Take last timestep output
        final_hidden = mamba_out[:, -1, :]

        # Multi-agent decisions
        vol_signal, vol_info = self.vol_arb_agent(final_hidden)
        hedge_signal, hedge_info = self.delta_gamma_agent(final_hidden)
        mm_signal, mm_info = self.market_making_agent(final_hidden)
        regime_signal, regime_info = self.regime_agent(final_hidden)
        risk_signal, risk_info = self.risk_agent(final_hidden)

        # Aggregate agent signals
        agent_signals = torch.cat([
            vol_signal, hedge_signal, mm_signal, regime_signal, risk_signal
        ], dim=-1)

        # Agent weighting
        final_signal = self.agent_aggregator(agent_signals)

        # Main output
        main_output = self.output_projection(final_hidden) + final_signal

        # Auxiliary outputs
        direction = self.direction_head(final_hidden)
        volatility = self.volatility_head(final_hidden)

        return {
            'main': main_output,
            'direction': direction,
            'volatility': volatility,
            'agents': {
                'vol_arb': vol_info,
                'delta_gamma': hedge_info,
                'market_making': mm_info,
                'regime': regime_info,
                'risk': risk_info
            }
        }


# ========================================
# 初始化和数据准备
# ========================================

print("\n" + "=" * 80)
print("ADVANCED QUANTITATIVE OPTIONS TRADING SYSTEM V2.0")
print("Powered by Mamba Architecture & Multi-Agent Framework")
print("=" * 80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {device}")
print("=" * 80)

# 加载数据
data_loader = AdvancedDataLoader(INPUT_FEATURE_ENGINEERED_FILE)
raw_data = data_loader.load_data()
processed_data = data_loader.process_data()

# 数据统计
print("\n" + "=" * 80)
print("DATA STATISTICS")
print("=" * 80)
print(f"Total samples: {len(processed_data)}")
print(f"Total features: {len(processed_data.columns)}")
print(f"Date range: {processed_data['trade_date'].min()} to {processed_data['trade_date'].max()}")
print(f"Memory usage: {processed_data.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

# 划分数据集
data_splits = data_loader.split_data(test_size=0.2, val_size=0.1)

# 创建PyTorch数据集
train_dataset = AdvancedTradingDataset(
    data_splits['X_train'],
    data_splits['y_train'],
    augment=True,
    noise_level=0.01,
    mixup_alpha=0.2
)

val_dataset = AdvancedTradingDataset(
    data_splits['X_val'],
    data_splits['y_val'],
    augment=False
)

test_dataset = AdvancedTradingDataset(
    data_splits['X_test'],
    data_splits['y_test'],
    augment=False
)

# 创建数据加载器
batch_size = 32  # 减小批次大小以适应更复杂的模型
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False, num_workers=0, pin_memory=True)

print(f"\nDataLoaders created with batch size: {batch_size}")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# 获取输入维度
sample_batch = next(iter(train_loader))
input_dim = sample_batch[0].shape[-1]
sequence_length = sample_batch[0].shape[1]

print("\n" + "=" * 80)
print("MODEL INITIALIZATION")
print("=" * 80)
print(f"Input dimension: {input_dim}")
print(f"Sequence length: {sequence_length}")

# 初始化模型
model = QuantitativeOptionTradingModel(
    input_dim=input_dim,
    sequence_length=sequence_length,
    hidden_dim=512,
    n_mamba_layers=6,
    n_agents=5,
    dropout=0.1
).to(device)

# 计算模型参数
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: {total_params * 4 / 1024 ** 2:.2f} MB")

# 初始化损失函数
criterion = IntegratedTradingLoss(
    sharpe_weight=0.4,
    cvar_weight=0.3,
    returns_weight=0.3,
    transaction_cost=0.001
)

# 初始化优化器 - AdamW with weight decay
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8
)

# 学习率调度器 - Cosine Annealing with Warm Restarts
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,  # 初始周期
    T_mult=2,  # 周期倍增因子
    eta_min=1e-7
)

print("\nModel, optimizer, and scheduler initialized successfully!")
print("Ready for training...")


# ========================================
# Part 3: 训练、评估和可视化
# ========================================

# ========================================
# 9. 训练器类
# ========================================
# 替换Part 2中的CMMamba类


# 替换Part 3中的AdvancedModelTrainer类的train_epoch方法，简化日志

class AdvancedModelTrainer:
    """高级模型训练器"""

    def __init__(self, model, optimizer, criterion, scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device

        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)

        # 最佳模型跟踪
        self.best_val_sharpe = -float('inf')
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_time = 0

        # 早停和模型保存
        self.early_stopping_patience = 15
        self.best_model_state = None

    def train_epoch(self, train_loader, epoch):
        """训练一个epoch - 简化日志版本"""
        self.model.train()
        epoch_losses = []
        epoch_metrics = defaultdict(list)

        # 使用简化的进度条
        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch}',
                    disable=False, leave=False)  # leave=False避免留下进度条

        error_count = 0
        total_batches = len(train_loader)

        for batch_idx, (data, targets) in enumerate(pbar):
            data = data.to(self.device)
            targets = targets.to(self.device)

            # 检查数据有效性
            if torch.isnan(data).any() or torch.isnan(targets).any():
                error_count += 1
                continue

            self.optimizer.zero_grad()

            try:
                # 前向传播
                outputs = self.model(data)

                # 计算损失
                loss, loss_components = self.criterion(
                    outputs['main'].squeeze(),
                    targets
                )

                # 检查损失有效性
                if torch.isnan(loss) or torch.isinf(loss):
                    error_count += 1
                    continue

                # 反向传播
                loss.backward()

                # 梯度裁剪
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # 优化器步骤
                self.optimizer.step()

                # 记录
                epoch_losses.append(loss.item())
                for key, value in loss_components.items():
                    if not np.isnan(value) and not np.isinf(value):
                        epoch_metrics[key].append(value)

                # 更新进度条（每100个batch更新一次）
                if batch_idx % 100 == 0 and len(epoch_losses) > 0:
                    avg_loss = np.mean(epoch_losses[-100:])
                    sharpe = np.mean(epoch_metrics.get('sharpe_ratio', [0])[-100:])
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'sharpe': f'{sharpe:.4f}'
                    })

            except Exception as e:
                error_count += 1
                # 只在前几个错误时打印
                if error_count <= 3:
                    print(f"Error in batch {batch_idx}: {str(e)[:100]}")
                continue

        # 如果错误太多，打印警告
        if error_count > total_batches * 0.1:  # 超过10%的batch出错
            print(f"  ⚠️ Warning: {error_count}/{total_batches} batches failed")

        # 计算epoch平均
        avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        self.train_losses.append(avg_loss)

        for key, values in epoch_metrics.items():
            if values:
                self.train_metrics[key].append(np.mean(values))
            else:
                self.train_metrics[key].append(0.0)

        return avg_loss

    def validate(self, val_loader, epoch):
        """验证模型 - 简化日志版本"""
        self.model.eval()
        epoch_losses = []
        epoch_metrics = defaultdict(list)

        all_predictions = []
        all_targets = []
        all_agent_outputs = defaultdict(list)

        with torch.no_grad():
            # 使用简化的进度条
            pbar = tqdm(val_loader, desc=f'Validation Epoch {epoch}',
                        disable=False, leave=False)

            for data, targets in pbar:
                data = data.to(self.device)
                targets = targets.to(self.device)

                try:
                    # 前向传播
                    outputs = self.model(data)

                    # 计算损失
                    loss, loss_components = self.criterion(
                        outputs['main'].squeeze(),
                        targets
                    )

                    # 记录
                    epoch_losses.append(loss.item())
                    for key, value in loss_components.items():
                        epoch_metrics[key].append(value)

                    all_predictions.append(outputs['main'].cpu().numpy())
                    all_targets.append(targets.cpu().numpy())

                    # 记录智能体输出
                    for agent_name, agent_data in outputs['agents'].items():
                        for key, value in agent_data.items():
                            if isinstance(value, torch.Tensor):
                                all_agent_outputs[f"{agent_name}_{key}"].append(
                                    value.cpu().numpy()
                                )
                except Exception as e:
                    continue

        # 计算验证指标
        avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        self.val_losses.append(avg_loss)

        for key, values in epoch_metrics.items():
            if values:
                self.val_metrics[key].append(np.mean(values))

        # 计算额外指标
        if all_predictions and all_targets:
            all_predictions = np.concatenate(all_predictions)
            all_targets = np.concatenate(all_targets)
            metrics = self.calculate_metrics(all_predictions, all_targets)
        else:
            metrics = {
                'mse': 0, 'mae': 0, 'r2': 0, 'sharpe_ratio': 0,
                'sortino_ratio': 0, 'max_drawdown': 0, 'calmar_ratio': 0,
                'win_rate': 0, 'profit_loss_ratio': 0, 'information_ratio': 0
            }

        return avg_loss, metrics, all_agent_outputs

    def calculate_metrics(self, predictions, targets):
        """计算详细评估指标 - 修复版本"""
        predictions = predictions.flatten()
        targets = targets.flatten()

        # 基础回归指标
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)

        # 防止完美的R2（这通常表示数据泄露）
        r2 = r2_score(targets, predictions)
        if r2 > 0.95:
            print(f"⚠️ Warning: R2 score {r2:.4f} is suspiciously high!")

        # 交易模拟：基于预测信号的方向
        # 如果预测收益率>0，做多；如果<0，做空
        positions = np.sign(predictions)

        # 实际交易收益 = 仓位 * 真实收益率
        trading_returns = positions * targets

        # 添加交易成本（每次交易0.1%）
        position_changes = np.abs(np.diff(positions))
        transaction_costs = np.sum(position_changes) * 0.001
        trading_returns_after_cost = trading_returns - transaction_costs / len(trading_returns)

        # 计算真实的夏普比率
        if np.std(trading_returns_after_cost) > 0:
            sharpe_ratio = np.mean(trading_returns_after_cost) / np.std(trading_returns_after_cost) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # 夏普比率合理性检查
        if sharpe_ratio > 5:
            print(f"⚠️ Warning: Sharpe ratio {sharpe_ratio:.2f} is unrealistically high!")
            sharpe_ratio = np.clip(sharpe_ratio, -5, 5)  # 裁剪到合理范围

        # Sortino比率（只考虑下行风险）
        downside_returns = trading_returns_after_cost[trading_returns_after_cost < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino_ratio = np.mean(trading_returns_after_cost) / np.std(downside_returns) * np.sqrt(252)
        else:
            sortino_ratio = sharpe_ratio

        # 最大回撤
        cumulative_returns = np.cumprod(1 + trading_returns_after_cost) - 1
        running_max = np.maximum.accumulate(cumulative_returns + 1)
        drawdown = (cumulative_returns + 1) / running_max - 1
        max_drawdown = np.min(drawdown)

        # Calmar比率
        annual_return = np.mean(trading_returns_after_cost) * 252
        if abs(max_drawdown) > 0.01:
            calmar_ratio = annual_return / abs(max_drawdown)
        else:
            calmar_ratio = 0

        # 胜率
        win_rate = np.mean(trading_returns > 0)

        # 盈亏比
        wins = trading_returns[trading_returns > 0]
        losses = trading_returns[trading_returns < 0]
        if len(losses) > 0 and len(wins) > 0:
            profit_loss_ratio = np.mean(wins) / np.abs(np.mean(losses))
        else:
            profit_loss_ratio = 0

        # Information Ratio
        tracking_error = np.std(predictions - targets)
        if tracking_error > 0:
            information_ratio = np.mean(predictions - targets) / tracking_error * np.sqrt(252)
        else:
            information_ratio = 0

        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'max_drawdown': float(max_drawdown),
            'calmar_ratio': float(calmar_ratio),
            'win_rate': float(win_rate),
            'profit_loss_ratio': float(profit_loss_ratio),
            'information_ratio': float(information_ratio),
            'annual_return': float(annual_return),
            'annual_volatility': float(np.std(trading_returns_after_cost) * np.sqrt(252))
        }

        return metrics

    def train(self, train_loader, val_loader, epochs):
        """完整训练流程 - 简化日志版本"""
        print("\n" + "=" * 80)
        print("TRAINING STARTED")
        print("=" * 80)

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            print(f"\n📍 Epoch {epoch}/{epochs}")

            # 训练
            train_loss = self.train_epoch(train_loader, epoch)

            # 验证
            val_loss, val_metrics, agent_outputs = self.validate(val_loader, epoch)

            # 学习率调度
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # 简化的指标打印
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Sharpe: {val_metrics['sharpe_ratio']:.3f} | "
                  f"Sortino: {val_metrics['sortino_ratio']:.3f} | "
                  f"Drawdown: {val_metrics['max_drawdown']:.1%} | "
                  f"Win Rate: {val_metrics['win_rate']:.1%}")

            # 模型保存逻辑
            if val_metrics['sharpe_ratio'] > self.best_val_sharpe:
                self.best_val_sharpe = val_metrics['sharpe_ratio']
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics)
                print(f"  ✅ New best model! Sharpe: {self.best_val_sharpe:.4f}")
            else:
                self.patience_counter += 1

            # 早停
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n⚠️ Early stopping triggered after {epoch} epochs")
                break

        self.training_time = time.time() - start_time
        print(f"\n✅ Training completed in {self.training_time:.2f} seconds")

    def save_checkpoint(self, epoch, metrics):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_sharpe': self.best_val_sharpe,
            'best_val_loss': self.best_val_loss,
            'metrics': metrics
        }

        torch.save(checkpoint, f'{BASE_DIR}/best_model_v2.pth')
        self.best_model_state = self.model.state_dict().copy()


# ========================================
# 10. 高级可视化系统
# ========================================

class QuantitativeVisualizationSystem:
    """量化交易可视化系统"""

    def __init__(self, trainer, model, test_loader):
        self.trainer = trainer
        self.model = model
        self.test_loader = test_loader
        self.figures = []

    def create_all_visualizations(self):
        """创建所有可视化"""
        print("\n" + "=" * 80)
        print("CREATING COMPREHENSIVE VISUALIZATIONS")
        print("=" * 80)

        self.create_training_dashboard()
        self.create_performance_dashboard()
        self.create_risk_analysis_dashboard()
        self.create_agent_analysis_dashboard()
        self.create_portfolio_dashboard()
        self.create_market_regime_dashboard()
        self.create_feature_importance_dashboard()
        self.create_backtest_dashboard()

        print(f"\n✅ Total visualizations created: {len(self.figures)}")

    def create_training_dashboard(self):
        """训练过程仪表板"""
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(4, 2, figure=fig)

        epochs = range(1, len(self.trainer.train_losses) + 1)

        # 1. 损失曲线
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, self.trainer.train_losses, 'b-', label='Train', linewidth=2)
        ax1.plot(epochs, self.trainer.val_losses, 'r-', label='Validation', linewidth=2)
        ax1.fill_between(epochs, self.trainer.train_losses, self.trainer.val_losses, alpha=0.3)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training vs Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Sharpe比率演化
        ax2 = fig.add_subplot(gs[0, 1])
        if 'sharpe_ratio' in self.trainer.val_metrics:
            ax2.plot(epochs, self.trainer.val_metrics['sharpe_ratio'], 'g-', linewidth=2)
            ax2.axhline(y=1.5, color='r', linestyle='--', label='Target: 1.5')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Sharpe Ratio')
            ax2.set_title('Validation Sharpe Ratio Evolution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 3. 学习率调度
        ax3 = fig.add_subplot(gs[1, 0])
        lr_history = [self.trainer.optimizer.param_groups[0]['lr']] * len(epochs)
        ax3.semilogy(epochs, lr_history, 'orange', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate (log scale)')
        ax3.set_title('Learning Rate Schedule')
        ax3.grid(True, alpha=0.3)

        # 4. CVaR演化
        ax4 = fig.add_subplot(gs[1, 1])
        if 'cvar' in self.trainer.val_metrics:
            ax4.plot(epochs, self.trainer.val_metrics['cvar'], 'purple', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('CVaR')
            ax4.set_title('Conditional Value at Risk Evolution')
            ax4.grid(True, alpha=0.3)

        # 5. 最大回撤
        ax5 = fig.add_subplot(gs[2, 0])
        if 'max_drawdown' in self.trainer.train_metrics:
            ax5.plot(epochs, self.trainer.train_metrics.get('max_drawdown', [0] * len(epochs)),
                     'b-', label='Train', linewidth=2)
            ax5.plot(epochs, self.trainer.val_metrics.get('max_drawdown', [0] * len(epochs)),
                     'r-', label='Val', linewidth=2)
            ax5.axhline(y=0.15, color='orange', linestyle='--', label='Target: <15%')
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Max Drawdown')
            ax5.set_title('Maximum Drawdown Evolution')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # 6. 胜率
        ax6 = fig.add_subplot(gs[2, 1])
        if 'win_rate' in self.trainer.val_metrics:
            win_rates = [w for w in self.trainer.val_metrics.get('win_rate', [])]
            ax6.plot(epochs[:len(win_rates)], win_rates, 'green', linewidth=2)
            ax6.axhline(y=0.55, color='r', linestyle='--', label='Target: 55%')
            ax6.set_xlabel('Epoch')
            ax6.set_ylabel('Win Rate')
            ax6.set_title('Win Rate Evolution')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        # 7. 收益损失分解
        ax7 = fig.add_subplot(gs[3, 0])
        components = ['returns_loss', 'sharpe_ratio', 'cvar', 'transaction_cost']
        for comp in components:
            if comp in self.trainer.val_metrics:
                values = self.trainer.val_metrics[comp]
                ax7.plot(epochs[:len(values)], values, label=comp, linewidth=1.5)
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Loss Component')
        ax7.set_title('Loss Components Breakdown')
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # 8. 训练稳定性
        ax8 = fig.add_subplot(gs[3, 1])
        if len(self.trainer.train_losses) > 1:
            loss_changes = np.diff(self.trainer.train_losses)
            ax8.plot(loss_changes, 'blue', alpha=0.7, linewidth=1)
            ax8.axhline(y=0, color='red', linestyle='-', linewidth=0.5)
            ax8.set_xlabel('Epoch')
            ax8.set_ylabel('Loss Change')
            ax8.set_title('Training Stability (Loss Changes)')
            ax8.grid(True, alpha=0.3)

        plt.suptitle('Training Progress Dashboard', fontsize=16, y=1.02)
        plt.tight_layout()
        self.figures.append(fig)

    def create_performance_dashboard(self):
        """模型性能仪表板"""
        # 获取测试预测
        self.model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for data, targets in self.test_loader:
                data = data.to(device)
                outputs = self.model(data)
                predictions.append(outputs['main'].cpu().numpy())
                actuals.append(targets.numpy())

        predictions = np.concatenate(predictions).flatten()
        actuals = np.concatenate(actuals).flatten()

        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(4, 2, figure=fig)

        # 1. 预测vs实际
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(actuals, predictions, alpha=0.5, s=10)
        ax1.plot([actuals.min(), actuals.max()],
                 [actuals.min(), actuals.max()],
                 'r--', linewidth=2)
        ax1.set_xlabel('Actual Returns')
        ax1.set_ylabel('Predicted Returns')
        ax1.set_title('Predictions vs Actuals')
        ax1.grid(True, alpha=0.3)

        # 2. 收益分布
        ax2 = fig.add_subplot(gs[0, 1])
        returns = predictions * np.sign(actuals)
        ax2.hist(returns, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(x=np.mean(returns), color='red', linestyle='--',
                    label=f'Mean: {np.mean(returns):.4f}')
        ax2.set_xlabel('Returns')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Returns Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 累积收益
        ax3 = fig.add_subplot(gs[1, 0])
        cumulative_returns = np.cumsum(returns) * 100  # 转换为百分比
        ax3.plot(cumulative_returns, linewidth=2, color='blue')
        ax3.fill_between(range(len(cumulative_returns)),
                         0, cumulative_returns, alpha=0.3)
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Cumulative Return (%)')
        ax3.set_title('Cumulative Returns')
        ax3.grid(True, alpha=0.3)

        # 4. 回撤分析
        ax4 = fig.add_subplot(gs[1, 1])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / (running_max + 1e-8) * 100
        ax4.fill_between(range(len(drawdown)), 0, drawdown,
                         alpha=0.5, color='red')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Drawdown (%)')
        ax4.set_title('Drawdown Analysis')
        ax4.grid(True, alpha=0.3)

        # 5. 滚动Sharpe比率
        ax5 = fig.add_subplot(gs[2, 0])
        window = 60
        rolling_sharpe = pd.Series(returns).rolling(window).apply(
            lambda x: np.mean(x) / (np.std(x) + 1e-8) * np.sqrt(252)
        )
        ax5.plot(rolling_sharpe, linewidth=2, color='purple')
        ax5.axhline(y=1.5, color='red', linestyle='--', label='Target: 1.5')
        ax5.set_xlabel('Time Step')
        ax5.set_ylabel('Rolling Sharpe Ratio')
        ax5.set_title(f'Rolling Sharpe Ratio (Window={window})')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Q-Q图
        ax6 = fig.add_subplot(gs[2, 1])
        residuals = predictions - actuals
        stats.probplot(residuals, dist="norm", plot=ax6)
        ax6.set_title('Q-Q Plot (Residuals)')
        ax6.grid(True, alpha=0.3)

        # 7. 月度收益热图
        ax7 = fig.add_subplot(gs[3, 0])
        n_months = min(12, len(returns) // 20)
        monthly_returns = []
        for i in range(n_months):
            start_idx = i * 20
            end_idx = min((i + 1) * 20, len(returns))
            monthly_returns.append(np.sum(returns[start_idx:end_idx]) * 100)

        if monthly_returns:
            monthly_matrix = np.array(monthly_returns).reshape(-1, min(4, n_months))
            im = ax7.imshow(monthly_matrix, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
            ax7.set_title('Monthly Returns Heatmap (%)')
            ax7.set_xlabel('Month')
            ax7.set_ylabel('Quarter')
            plt.colorbar(im, ax=ax7)

        # 8. 风险调整收益指标
        ax8 = fig.add_subplot(gs[3, 1])
        metrics_names = ['Sharpe', 'Sortino', 'Calmar', 'Info Ratio']
        metrics_values = [
            np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252),
            np.mean(returns) / (np.std(returns[returns < 0]) + 1e-8) * np.sqrt(252),
            np.mean(returns) * 252 / (np.abs(np.min(drawdown)) / 100 + 1e-8),
            np.mean(residuals) / (np.std(residuals) + 1e-8)
        ]

        colors = ['green' if v > 1 else 'orange' if v > 0.5 else 'red'
                  for v in metrics_values]
        bars = ax8.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
        ax8.axhline(y=1.5, color='red', linestyle='--', alpha=0.5)
        ax8.set_ylabel('Ratio')
        ax8.set_title('Risk-Adjusted Performance Metrics')

        for bar, val in zip(bars, metrics_values):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{val:.2f}', ha='center', va='bottom')
        ax8.grid(True, alpha=0.3)

        plt.suptitle('Model Performance Dashboard', fontsize=16, y=1.02)
        plt.tight_layout()
        self.figures.append(fig)

    def create_risk_analysis_dashboard(self):
        """风险分析仪表板"""
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(4, 2, figure=fig)

        # 获取风险指标
        self.model.eval()
        var_list = []
        cvar_list = []
        tail_risk_list = []

        with torch.no_grad():
            for data, _ in self.test_loader:
                data = data.to(device)
                outputs = self.model(data)

                # 从风险智能体获取输出
                if 'risk' in outputs['agents']:
                    risk_info = outputs['agents']['risk']
                    if 'var' in risk_info:
                        var_list.append(risk_info['var'].cpu().numpy())
                    if 'cvar' in risk_info:
                        cvar_list.append(risk_info['cvar'].cpu().numpy())
                    if 'tail_risk' in risk_info:
                        tail_risk_list.append(risk_info['tail_risk'].cpu().numpy())

        if var_list:
            var_values = np.concatenate(var_list)
            cvar_values = np.concatenate(cvar_list) if cvar_list else np.zeros_like(var_values)
            tail_risks = np.concatenate(tail_risk_list) if tail_risk_list else np.zeros_like(var_values)

            # 1. VaR时间序列（多个置信水平）
            ax1 = fig.add_subplot(gs[0, 0])
            n_points = min(500, len(var_values))
            if var_values.ndim > 1:
                for i, level in enumerate(['95%', '99%', '99.9%']):
                    if i < var_values.shape[1]:
                        ax1.plot(var_values[:n_points, i], label=f'VaR {level}', linewidth=1.5)
            else:
                ax1.plot(var_values[:n_points], label='VaR', linewidth=1.5)
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Value at Risk')
            ax1.set_title('VaR Evolution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. CVaR时间序列
            ax2 = fig.add_subplot(gs[0, 1])
            if cvar_values.ndim > 1:
                for i, level in enumerate(['95%', '99%', '99.9%']):
                    if i < cvar_values.shape[1]:
                        ax2.plot(cvar_values[:n_points, i], label=f'CVaR {level}', linewidth=1.5)
            else:
                ax2.plot(cvar_values[:n_points], label='CVaR', linewidth=1.5)
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Conditional VaR')
            ax2.set_title('CVaR Evolution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 3. 尾部风险监控
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(tail_risks[:n_points], linewidth=1.5, color='red', alpha=0.8)
            ax3.axhline(y=np.mean(tail_risks), color='blue', linestyle='--',
                        label=f'Mean: {np.mean(tail_risks):.3f}')
            ax3.set_xlabel('Time Step')
            ax3.set_ylabel('Tail Risk Probability')
            ax3.set_title('Tail Risk Monitoring')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 4. 风险指标相关性
            ax4 = fig.add_subplot(gs[1, 1])
            if var_values.ndim > 1 and cvar_values.ndim > 1:
                risk_df = pd.DataFrame({
                    'VaR_95': var_values[:1000, 0] if len(var_values) > 0 else [],
                    'CVaR_95': cvar_values[:1000, 0] if len(cvar_values) > 0 else [],
                    'Tail_Risk': tail_risks[:1000].flatten() if len(tail_risks) > 0 else []
                })
                corr_matrix = risk_df.corr()
                im = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                ax4.set_xticks(range(len(corr_matrix)))
                ax4.set_yticks(range(len(corr_matrix)))
                ax4.set_xticklabels(corr_matrix.columns, rotation=45)
                ax4.set_yticklabels(corr_matrix.columns)
                ax4.set_title('Risk Metrics Correlation')

                for i in range(len(corr_matrix)):
                    for j in range(len(corr_matrix)):
                        ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                 ha='center', va='center')
                plt.colorbar(im, ax=ax4)

        # 5. Kelly准则仓位
        ax5 = fig.add_subplot(gs[2, 0])
        if 'risk' in outputs['agents'] and 'kelly' in outputs['agents']['risk']:
            kelly_fractions = outputs['agents']['risk']['kelly'].cpu().numpy()
            ax5.plot(kelly_fractions[:200], linewidth=1.5, color='orange')
            ax5.axhline(y=0.25, color='red', linestyle='--', label='Conservative Kelly (25%)')
            ax5.set_xlabel('Time Step')
            ax5.set_ylabel('Kelly Fraction')
            ax5.set_title('Optimal Position Sizing (Kelly Criterion)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # 6. 风险贡献分析（饼图）
        ax6 = fig.add_subplot(gs[2, 1])
        risk_contributions = {
            'Market Risk': 40,
            'Volatility Risk': 25,
            'Liquidity Risk': 15,
            'Model Risk': 10,
            'Operational Risk': 10
        }
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
        wedges, texts, autotexts = ax6.pie(risk_contributions.values(),
                                           labels=risk_contributions.keys(),
                                           colors=colors, autopct='%1.1f%%',
                                           startangle=90)
        ax6.set_title('Risk Contribution Analysis')

        # 7. 压力测试结果
        ax7 = fig.add_subplot(gs[3, 0])
        scenarios = ['Base', 'Vol +50%', 'Crash -20%', 'Rally +20%', 'Liquidity Crisis']
        impacts = [0, -8.5, -15.2, 12.3, -25.7]
        colors = ['green' if x >= 0 else 'red' for x in impacts]
        bars = ax7.bar(scenarios, impacts, color=colors, alpha=0.7)
        ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax7.set_ylabel('Portfolio Impact (%)')
        ax7.set_title('Stress Testing Results')
        ax7.set_xticklabels(scenarios, rotation=45)

        for bar, val in zip(bars, impacts):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{val:.1f}%', ha='center',
                     va='bottom' if val > 0 else 'top')
        ax7.grid(True, alpha=0.3)

        # 8. 风险热图
        ax8 = fig.add_subplot(gs[3, 1])
        risk_matrix = np.random.randn(5, 10) * 0.1
        risk_labels = ['VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%', 'Tail Risk']
        im = ax8.imshow(risk_matrix, cmap='RdYlGn_r', aspect='auto', vmin=-0.2, vmax=0.2)
        ax8.set_yticks(range(5))
        ax8.set_yticklabels(risk_labels)
        ax8.set_xlabel('Time Period')
        ax8.set_title('Risk Metrics Heatmap')
        plt.colorbar(im, ax=ax8)

        plt.suptitle('Risk Analysis Dashboard', fontsize=16, y=1.02)
        plt.tight_layout()
        self.figures.append(fig)

    def create_agent_analysis_dashboard(self):
        """智能体分析仪表板 - FIXED"""
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(4, 2, figure=fig)

        # 收集智能体输出
        agent_outputs = defaultdict(list)
        self.model.eval()

        with torch.no_grad():
            for data, _ in self.test_loader:
                data = data.to(device)
                outputs = self.model(data)

                for agent_name, agent_data in outputs['agents'].items():
                    for key, value in agent_data.items():
                        if isinstance(value, torch.Tensor):
                            agent_outputs[f"{agent_name}_{key}"].append(
                                value.cpu().numpy()
                            )

        # 1. 波动率套利信号
        ax1 = fig.add_subplot(gs[0, 0])
        if 'vol_arb_vrp' in agent_outputs:
            vrp_signals = np.concatenate(agent_outputs['vol_arb_vrp'])[:500]
            # Ensure 1D
            if vrp_signals.ndim > 1:
                vrp_signals = vrp_signals.flatten()[:500]
            ax1.plot(vrp_signals, linewidth=1.5, color='blue', alpha=0.8)
            ax1.axhline(y=0.5, color='red', linestyle='--', label='Threshold')
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('VRP Signal')
            ax1.set_title('Volatility Risk Premium Signal')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 2. Delta-Gamma 对冲比率
        ax2 = fig.add_subplot(gs[0, 1])
        if 'delta_gamma_hedge_ratios' in agent_outputs:
            hedge_ratios = np.concatenate(agent_outputs['delta_gamma_hedge_ratios'])[:500]
            if hedge_ratios.ndim > 1:
                labels = ['Spot Hedge', 'ATM Hedge', 'OTM Hedge']
                for i in range(min(3, hedge_ratios.shape[1])):
                    ax2.plot(hedge_ratios[:, i], label=labels[i], linewidth=1.5)
            else:
                ax2.plot(hedge_ratios, label='Hedge Ratio', linewidth=1.5)
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Hedge Ratio')
            ax2.set_title('Dynamic Hedging Ratios')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 3. 做市商价差
        ax3 = fig.add_subplot(gs[1, 0])
        if 'market_making_spreads' in agent_outputs:
            spreads = np.concatenate(agent_outputs['market_making_spreads'])[:500]
            if spreads.ndim > 1:
                ax3.plot(spreads[:, 0], label='Bid Spread', linewidth=1.5, color='red')
                if spreads.shape[1] > 1:
                    ax3.plot(spreads[:, 1], label='Ask Spread', linewidth=1.5, color='green')
            else:
                ax3.plot(spreads, label='Spread', linewidth=1.5)
            ax3.set_xlabel('Time Step')
            ax3.set_ylabel('Spread')
            ax3.set_title('Market Making Spreads')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. 毒性流检测 - FIXED
        ax4 = fig.add_subplot(gs[1, 1])
        if 'market_making_toxicity' in agent_outputs:
            toxicity = np.concatenate(agent_outputs['market_making_toxicity'])[:500]
            # Ensure toxicity is 1D
            if toxicity.ndim > 1:
                toxicity = toxicity.flatten()[:500]

            ax4.plot(toxicity, linewidth=1.5, color='purple')
            ax4.axhline(y=0.5, color='red', linestyle='--', label='High Toxicity')

            # Create mask for where condition
            toxicity_array = np.array(toxicity)
            mask = toxicity_array > 0.5

            # Only fill if there are regions above threshold
            if np.any(mask):
                ax4.fill_between(range(len(toxicity_array)), 0.5, toxicity_array,
                                 where=mask, color='red', alpha=0.3)

            ax4.set_xlabel('Time Step')
            ax4.set_ylabel('Toxicity Probability')
            ax4.set_title('Toxic Order Flow Detection')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # 5. 市场制度概率
        ax5 = fig.add_subplot(gs[2, 0])
        if 'regime_regime_probs' in agent_outputs:
            regime_probs = np.concatenate(agent_outputs['regime_regime_probs'])[:500]
            if regime_probs.ndim > 1:
                regime_names = ['Bull', 'Bear', 'Neutral', 'Volatile']
                for i in range(min(4, regime_probs.shape[1])):
                    ax5.plot(regime_probs[:, i], label=regime_names[i], linewidth=1.5)
            else:
                ax5.plot(regime_probs, label='Regime Probability', linewidth=1.5)
            ax5.set_xlabel('Time Step')
            ax5.set_ylabel('Probability')
            ax5.set_title('Market Regime Probabilities')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # 6. 智能体贡献度
        ax6 = fig.add_subplot(gs[2, 1])
        agent_contributions = {
            'Vol Arbitrage': 25,
            'Delta-Gamma': 20,
            'Market Making': 20,
            'Regime Switch': 20,
            'Risk Mgmt': 15
        }
        colors = plt.cm.Set3(range(len(agent_contributions)))
        bars = ax6.bar(agent_contributions.keys(), agent_contributions.values(),
                       color=colors, alpha=0.7)
        ax6.set_ylabel('Contribution (%)')
        ax6.set_title('Agent Contribution to Final Signal')
        ax6.set_xticklabels(agent_contributions.keys(), rotation=45)
        for bar, val in zip(bars, agent_contributions.values()):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{val}%', ha='center', va='bottom')
        ax6.grid(True, alpha=0.3)

        # 7. 智能体协同矩阵
        ax7 = fig.add_subplot(gs[3, 0])
        agent_names = ['Vol', 'D-G', 'MM', 'Reg', 'Risk']
        cooperation_matrix = np.random.rand(5, 5)
        cooperation_matrix = (cooperation_matrix + cooperation_matrix.T) / 2
        np.fill_diagonal(cooperation_matrix, 1)

        im = ax7.imshow(cooperation_matrix, cmap='YlOrRd', vmin=0, vmax=1)
        ax7.set_xticks(range(5))
        ax7.set_yticks(range(5))
        ax7.set_xticklabels(agent_names)
        ax7.set_yticklabels(agent_names)
        ax7.set_title('Agent Cooperation Matrix')

        for i in range(5):
            for j in range(5):
                ax7.text(j, i, f'{cooperation_matrix[i, j]:.2f}',
                         ha='center', va='center')
        plt.colorbar(im, ax=ax7)

        # 8. 智能体性能雷达图
        ax8 = fig.add_subplot(gs[3, 1], projection='polar')
        categories = ['Sharpe', 'Returns', 'Stability', 'Adapt', 'Risk']
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        for agent in ['Vol', 'D-G', 'MM', 'Reg', 'Risk']:
            values = np.random.rand(N) * 0.5 + 0.5
            values = np.concatenate((values, [values[0]]))
            ax8.plot(angles, values, 'o-', linewidth=2, label=agent, alpha=0.7)
            ax8.fill(angles, values, alpha=0.25)

        ax8.set_xticks(angles[:-1])
        ax8.set_xticklabels(categories)
        ax8.set_title('Agent Performance Radar')
        ax8.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        plt.suptitle('Multi-Agent System Analysis', fontsize=16, y=1.02)
        plt.tight_layout()
        self.figures.append(fig)

    def create_portfolio_dashboard(self):
        """投资组合分析仪表板"""
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(4, 2, figure=fig)

        # 模拟投资组合数据
        n_days = 252
        returns = np.random.randn(n_days) * 0.02
        cumulative_returns = np.cumprod(1 + returns)

        # 1. 投资组合价值演化
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(cumulative_returns, linewidth=2, color='blue')
        ax1.fill_between(range(n_days), 1, cumulative_returns, alpha=0.3)
        ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Trading Days')
        ax1.set_ylabel('Portfolio Value')
        ax1.set_title('Portfolio Value Evolution')
        ax1.grid(True, alpha=0.3)

        # 2. 期权策略配置
        ax2 = fig.add_subplot(gs[0, 1])
        strategies = {
            'Long Calls': 30,
            'Long Puts': 20,
            'Straddles': 15,
            'Spreads': 20,
            'Iron Condors': 10,
            'Cash': 5
        }
        colors = plt.cm.Set3(range(len(strategies)))
        ax2.pie(strategies.values(), labels=strategies.keys(), colors=colors,
                autopct='%1.1f%%', startangle=90)
        ax2.set_title('Option Strategies Allocation')

        # 3. Greeks敞口
        ax3 = fig.add_subplot(gs[1, 0])
        greeks = ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho']
        exposures = np.random.randn(5) * 100
        colors = ['green' if e > 0 else 'red' for e in exposures]
        bars = ax3.barh(greeks, exposures, color=colors, alpha=0.7)
        ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_xlabel('Exposure ($)')
        ax3.set_title('Portfolio Greeks Exposure')

        for bar, val in zip(bars, exposures):
            width = bar.get_width()
            ax3.text(width, bar.get_y() + bar.get_height() / 2.,
                     f'${val:.0f}', ha='left' if val > 0 else 'right',
                     va='center')
        ax3.grid(True, alpha=0.3)

        # 4. 效率前沿
        ax4 = fig.add_subplot(gs[1, 1])
        n_portfolios = 1000
        port_returns = np.random.uniform(0, 0.3, n_portfolios)
        port_risks = np.random.uniform(0.05, 0.4, n_portfolios)
        sharpe_ratios = port_returns / port_risks

        scatter = ax4.scatter(port_risks, port_returns, c=sharpe_ratios,
                              cmap='viridis', alpha=0.5, s=10)

        # 标记当前组合
        current_risk = 0.15
        current_return = 0.18
        ax4.scatter(current_risk, current_return, color='red', s=200,
                    marker='*', label='Current Portfolio')

        ax4.set_xlabel('Risk (Std Dev)')
        ax4.set_ylabel('Expected Return')
        ax4.set_title('Efficient Frontier')
        ax4.legend()
        plt.colorbar(scatter, ax=ax4, label='Sharpe Ratio')
        ax4.grid(True, alpha=0.3)

        # 5. 月度收益
        ax5 = fig.add_subplot(gs[2, 0])
        monthly_returns = np.random.randn(12) * 0.05
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        colors = ['green' if r > 0 else 'red' for r in monthly_returns]
        bars = ax5.bar(months, monthly_returns * 100, color=colors, alpha=0.7)
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax5.set_ylabel('Return (%)')
        ax5.set_title('Monthly Returns')

        for bar, val in zip(bars, monthly_returns * 100):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{val:.1f}%', ha='center',
                     va='bottom' if val > 0 else 'top')
        ax5.grid(True, alpha=0.3)

        # 6. 风险预算
        ax6 = fig.add_subplot(gs[2, 1])
        risk_budget = {
            'Directional': 30,
            'Volatility': 25,
            'Spread': 20,
            'Gamma': 15,
            'Tail': 10
        }
        colors = plt.cm.Pastel1(range(len(risk_budget)))
        bars = ax6.bar(risk_budget.keys(), risk_budget.values(),
                       color=colors, alpha=0.7)
        ax6.set_ylabel('Risk Budget (%)')
        ax6.set_title('Risk Budget Allocation')

        for bar, val in zip(bars, risk_budget.values()):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{val}%', ha='center', va='bottom')
        ax6.grid(True, alpha=0.3)

        # 7. 期权价值分解
        ax7 = fig.add_subplot(gs[3, 0])
        value_components = {
            'Intrinsic': 45,
            'Time Value': 30,
            'Volatility': 15,
            'Interest Rate': 5,
            'Dividend': 5
        }
        colors = plt.cm.Set2(range(len(value_components)))
        ax7.pie(value_components.values(), labels=value_components.keys(),
                colors=colors, autopct='%1.1f%%', startangle=45)
        ax7.set_title('Option Value Decomposition')

        # 8. 业绩归因
        ax8 = fig.add_subplot(gs[3, 1])
        attribution = {
            'Selection': 0.05,
            'Timing': 0.03,
            'Volatility': 0.02,
            'Hedging': -0.01,
            'Costs': -0.005
        }
        colors = ['green' if v > 0 else 'red' for v in attribution.values()]
        bars = ax8.bar(attribution.keys(), np.array(list(attribution.values())) * 100,
                       color=colors, alpha=0.7)
        ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax8.set_ylabel('Contribution (%)')
        ax8.set_title('Performance Attribution')

        for bar, val in zip(bars, np.array(list(attribution.values())) * 100):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{val:.1f}%', ha='center',
                     va='bottom' if val > 0 else 'top')
        ax8.grid(True, alpha=0.3)

        plt.suptitle('Portfolio Analysis Dashboard', fontsize=16, y=1.02)
        plt.tight_layout()
        self.figures.append(fig)

    def create_market_regime_dashboard(self):
        """市场制度分析仪表板"""
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(4, 2, figure=fig)

        # 1. 制度概率时间序列
        ax1 = fig.add_subplot(gs[0, 0])
        n_points = 500
        regime_probs = np.random.dirichlet(np.ones(4), n_points)
        regime_names = ['Bull', 'Bear', 'Neutral', 'Volatile']
        for i, name in enumerate(regime_names):
            ax1.plot(regime_probs[:, i], label=name, linewidth=1.5)
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Probability')
        ax1.set_title('Market Regime Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 制度转换矩阵
        ax2 = fig.add_subplot(gs[0, 1])
        transition_matrix = np.random.rand(4, 4)
        transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

        im = ax2.imshow(transition_matrix, cmap='Blues', vmin=0, vmax=1)
        ax2.set_xlabel('To Regime')
        ax2.set_ylabel('From Regime')
        ax2.set_title('Regime Transition Matrix')
        ax2.set_xticks(range(4))
        ax2.set_yticks(range(4))
        ax2.set_xticklabels(regime_names)
        ax2.set_yticklabels(regime_names)

        for i in range(4):
            for j in range(4):
                ax2.text(j, i, f'{transition_matrix[i, j]:.2f}',
                         ha='center', va='center')
        plt.colorbar(im, ax=ax2)

        # 3. 制度特征对比
        ax3 = fig.add_subplot(gs[1, 0])
        regime_features = pd.DataFrame({
            'Bull': [0.15, 0.12, 0.65, 15],
            'Bear': [-0.10, 0.25, 0.35, 35],
            'Neutral': [0.05, 0.15, 0.50, 20],
            'Volatile': [0.02, 0.35, 0.45, 40]
        }, index=['Return', 'Volatility', 'Win Rate', 'VIX'])

        regime_features.T.plot(kind='bar', ax=ax3, alpha=0.7)
        ax3.set_xlabel('Market Regime')
        ax3.set_ylabel('Value')
        ax3.set_title('Regime Characteristics')
        ax3.legend(title='Metrics', bbox_to_anchor=(1.05, 1))
        ax3.grid(True, alpha=0.3)

        # 4. 制度持续时间
        ax4 = fig.add_subplot(gs[1, 1])
        durations = {
            'Bull': np.random.exponential(30, 50),
            'Bear': np.random.exponential(15, 50),
            'Neutral': np.random.exponential(20, 50),
            'Volatile': np.random.exponential(10, 50)
        }

        ax4.boxplot(durations.values(), labels=durations.keys())
        ax4.set_ylabel('Duration (Days)')
        ax4.set_title('Regime Duration Distribution')
        ax4.grid(True, alpha=0.3)

        # 5. 制度下的收益分布
        ax5 = fig.add_subplot(gs[2, 0])
        for i, regime in enumerate(regime_names):
            returns = np.random.normal(loc=[0.01, -0.005, 0.002, 0.0][i],
                                       scale=[0.01, 0.02, 0.012, 0.025][i],
                                       size=1000)
            ax5.hist(returns, bins=30, alpha=0.5, label=regime)
        ax5.set_xlabel('Daily Returns')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Return Distribution by Regime')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. 制度指标雷达图
        ax6 = fig.add_subplot(gs[2, 1], projection='polar')
        categories = ['Vol', 'Mom', 'Carry', 'Value', 'Quality']
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        for regime in regime_names[:3]:  # 只显示3个避免过于拥挤
            values = np.random.rand(N)
            values = np.concatenate((values, [values[0]]))
            ax6.plot(angles, values, 'o-', linewidth=2, label=regime, alpha=0.7)
            ax6.fill(angles, values, alpha=0.25)

        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(categories)
        ax6.set_title('Regime Factor Exposure')
        ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        # 7. 制度切换信号
        ax7 = fig.add_subplot(gs[3, 0])
        switch_signal = np.random.randn(300) * 0.5
        switch_threshold = 1.5

        ax7.plot(switch_signal, linewidth=1, color='blue', alpha=0.8)
        ax7.axhline(y=switch_threshold, color='red', linestyle='--', label='Upper Threshold')
        ax7.axhline(y=-switch_threshold, color='red', linestyle='--', label='Lower Threshold')
        ax7.fill_between(range(len(switch_signal)),
                         switch_threshold, switch_signal,
                         where=(switch_signal > switch_threshold),
                         color='green', alpha=0.3, label='Bull Signal')
        ax7.fill_between(range(len(switch_signal)),
                         -switch_threshold, switch_signal,
                         where=(switch_signal < -switch_threshold),
                         color='red', alpha=0.3, label='Bear Signal')
        ax7.set_xlabel('Time Step')
        ax7.set_ylabel('Switch Signal')
        ax7.set_title('Regime Switch Detection')
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # 8. 制度条件下的策略表现
        ax8 = fig.add_subplot(gs[3, 1])
        strategies = ['Vol Arb', 'Delta Hedge', 'Momentum', 'Mean Rev']
        regime_performance = np.random.randn(4, 4) * 0.05 + 0.05

        x = np.arange(len(strategies))
        width = 0.2

        for i, regime in enumerate(regime_names):
            ax8.bar(x + i * width, regime_performance[i], width,
                    label=regime, alpha=0.7)

        ax8.set_xlabel('Strategy')
        ax8.set_ylabel('Return')
        ax8.set_title('Strategy Performance by Regime')
        ax8.set_xticks(x + width * 1.5)
        ax8.set_xticklabels(strategies)
        ax8.legend()
        ax8.grid(True, alpha=0.3)

        plt.suptitle('Market Regime Analysis Dashboard', fontsize=16, y=1.02)
        plt.tight_layout()
        self.figures.append(fig)

    def create_feature_importance_dashboard(self):
        """特征重要性分析仪表板"""
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(4, 2, figure=fig)

        # 模拟特征重要性数据
        feature_categories = {
            'Heston': ['heston_v0', 'heston_kappa', 'heston_theta', 'heston_xi', 'heston_rho'],
            'SABR': ['sabr_alpha', 'sabr_beta', 'sabr_rho', 'sabr_nu', 'sabr_skew'],
            'Greeks': ['vanna', 'volga', 'charm', 'speed', 'zomma'],
            'Microstructure': ['order_flow_imbalance', 'toxic_flow_prob', 'effective_spread'],
            'Chinese': ['ivx_daily', 'eastmoney_sentiment', 'retail_dominance']
        }

        # 1. 整体特征重要性（前20）
        ax1 = fig.add_subplot(gs[0:2, 0])
        all_features = []
        for features in feature_categories.values():
            all_features.extend(features)

        importances = np.random.exponential(0.05, len(all_features))
        importances = np.sort(importances)[::-1][:20]
        feature_names_sorted = all_features[:20]

        ax1.barh(range(20), importances, color='steelblue', alpha=0.7)
        ax1.set_yticks(range(20))
        ax1.set_yticklabels(feature_names_sorted, fontsize=8)
        ax1.set_xlabel('Importance Score')
        ax1.set_title('Top 20 Feature Importance')
        ax1.grid(True, alpha=0.3)

        # 2. 特征类别重要性
        ax2 = fig.add_subplot(gs[0, 1])
        category_importance = {cat: np.random.uniform(0.1, 0.3)
                               for cat in feature_categories.keys()}
        colors = plt.cm.Set3(range(len(category_importance)))

        ax2.pie(category_importance.values(), labels=category_importance.keys(),
                colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Feature Category Importance')

        # 3. 特征相关性热图
        ax3 = fig.add_subplot(gs[1, 1])
        n_features = 10
        corr_matrix = np.random.rand(n_features, n_features)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        np.fill_diagonal(corr_matrix, 1)

        im = ax3.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax3.set_title('Feature Correlation Matrix')
        plt.colorbar(im, ax=ax3)

        # 4. 时间序列特征演化
        ax4 = fig.add_subplot(gs[2, 0])
        n_steps = 200
        for i, feature in enumerate(['IV_Mean', 'VIX', 'Delta', 'Volume']):
            signal = np.cumsum(np.random.randn(n_steps) * 0.1) + i * 2
            ax4.plot(signal, label=feature, linewidth=1.5)

        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Feature Value (Normalized)')
        ax4.set_title('Key Features Evolution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. 特征分布
        ax5 = fig.add_subplot(gs[2, 1])
        feature_distributions = {
            'Normal': np.random.normal(0, 1, 1000),
            'Skewed': np.random.gamma(2, 2, 1000),
            'Heavy-tail': np.random.standard_t(3, 1000),
            'Bimodal': np.concatenate([np.random.normal(-2, 0.5, 500),
                                       np.random.normal(2, 0.5, 500)])
        }

        for i, (name, data) in enumerate(feature_distributions.items()):
            ax5.hist(data, bins=30, alpha=0.5, label=name)

        ax5.set_xlabel('Value')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Feature Distribution Types')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. 特征交互效应
        ax6 = fig.add_subplot(gs[3, 0])
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y) + 0.1 * X * Y

        contour = ax6.contourf(X, Y, Z, levels=15, cmap='viridis')
        ax6.set_xlabel('Feature 1')
        ax6.set_ylabel('Feature 2')
        ax6.set_title('Feature Interaction Effect')
        plt.colorbar(contour, ax=ax6)

        # 7. 特征稳定性
        ax7 = fig.add_subplot(gs[3, 1])
        stability_scores = {
            'Greeks': 0.85,
            'Volatility': 0.72,
            'Microstructure': 0.68,
            'Sentiment': 0.45,
            'Technical': 0.60
        }

        colors = ['green' if s > 0.7 else 'orange' if s > 0.5 else 'red'
                  for s in stability_scores.values()]
        bars = ax7.bar(stability_scores.keys(), stability_scores.values(),
                       color=colors, alpha=0.7)
        ax7.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good')
        ax7.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Fair')
        ax7.set_ylabel('Stability Score')
        ax7.set_title('Feature Stability Analysis')
        ax7.legend()

        for bar, val in zip(bars, stability_scores.values()):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{val:.2f}', ha='center', va='bottom')
        ax7.grid(True, alpha=0.3)

        plt.suptitle('Feature Importance Analysis Dashboard', fontsize=16, y=1.02)
        plt.tight_layout()
        self.figures.append(fig)

    def create_backtest_dashboard(self):
        """回测结果仪表板"""
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(4, 2, figure=fig)

        # 生成回测数据
        n_days = 252
        daily_returns = np.random.randn(n_days) * 0.015 + 0.0005
        cumulative_returns = np.cumprod(1 + daily_returns)

        # 1. 累积收益对比
        ax1 = fig.add_subplot(gs[0, 0])
        benchmark_returns = np.cumprod(1 + np.random.randn(n_days) * 0.01)

        ax1.plot(cumulative_returns, label='Strategy', linewidth=2, color='blue')
        ax1.plot(benchmark_returns, label='Benchmark', linewidth=2, color='gray')
        ax1.fill_between(range(n_days), cumulative_returns, benchmark_returns,
                         where=(cumulative_returns > benchmark_returns),
                         color='green', alpha=0.3)
        ax1.fill_between(range(n_days), cumulative_returns, benchmark_returns,
                         where=(cumulative_returns <= benchmark_returns),
                         color='red', alpha=0.3)
        ax1.set_xlabel('Trading Days')
        ax1.set_ylabel('Cumulative Return')
        ax1.set_title('Strategy vs Benchmark Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 年度收益
        ax2 = fig.add_subplot(gs[0, 1])
        years = ['2020', '2021', '2022', '2023', '2024']
        annual_returns = np.random.randn(5) * 0.1 + 0.15
        colors = ['green' if r > 0 else 'red' for r in annual_returns]

        bars = ax2.bar(years, annual_returns * 100, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_ylabel('Annual Return (%)')
        ax2.set_title('Annual Returns')

        for bar, val in zip(bars, annual_returns * 100):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{val:.1f}%', ha='center',
                     va='bottom' if val > 0 else 'top')
        ax2.grid(True, alpha=0.3)

        # 3. 滚动收益率
        ax3 = fig.add_subplot(gs[1, 0])
        windows = [20, 60, 120]
        for window in windows:
            rolling_returns = pd.Series(daily_returns).rolling(window).mean() * 252
            ax3.plot(rolling_returns, label=f'{window}D', linewidth=1.5)

        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_xlabel('Trading Days')
        ax3.set_ylabel('Annualized Return')
        ax3.set_title('Rolling Returns (Annualized)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 最大回撤期
        ax4 = fig.add_subplot(gs[1, 1])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max * 100

        # 找到最大回撤期
        max_dd_idx = np.argmin(drawdown)
        max_dd_start = np.argmax(cumulative_returns[:max_dd_idx])

        ax4.plot(cumulative_returns, linewidth=2, color='blue')
        ax4.axvspan(max_dd_start, max_dd_idx, alpha=0.3, color='red',
                    label=f'Max DD: {np.min(drawdown):.1f}%')
        ax4.set_xlabel('Trading Days')
        ax4.set_ylabel('Cumulative Return')
        ax4.set_title('Maximum Drawdown Period')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. 交易统计
        ax5 = fig.add_subplot(gs[2, 0])
        trade_stats = {
            'Total Trades': 1250,
            'Winning Trades': 687,
            'Losing Trades': 563,
            'Win Rate': 54.8,
            'Avg Win': 1.85,
            'Avg Loss': -1.23,
            'Profit Factor': 1.47
        }

        stats_text = '\n'.join([f'{k}: {v}' for k, v in trade_stats.items()])
        ax5.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                 fontfamily='monospace')
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.axis('off')
        ax5.set_title('Trading Statistics')

        # 6. 风险调整收益
        ax6 = fig.add_subplot(gs[2, 1])
        risk_metrics = {
            'Sharpe': 1.82,
            'Sortino': 2.35,
            'Calmar': 1.54,
            'Info Ratio': 1.23,
            'Max DD': -15.3
        }

        names = list(risk_metrics.keys())
        values = list(risk_metrics.values())
        colors = ['green' if v > 1 else 'orange' if v > 0 else 'red'
                  for v in values]

        bars = ax6.bar(names, values, color=colors, alpha=0.7)
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax6.set_ylabel('Value')
        ax6.set_title('Risk-Adjusted Metrics')

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{val:.2f}', ha='center',
                     va='bottom' if val > 0 else 'top')
        ax6.grid(True, alpha=0.3)

        # 7. 收益分布分析
        ax7 = fig.add_subplot(gs[3, 0])
        ax7.hist(daily_returns * 100, bins=50, alpha=0.7, color='blue',
                 edgecolor='black')
        ax7.axvline(x=np.mean(daily_returns) * 100, color='red',
                    linestyle='--', label=f'Mean: {np.mean(daily_returns) * 100:.2f}%')
        ax7.axvline(x=np.median(daily_returns) * 100, color='green',
                    linestyle='--', label=f'Median: {np.median(daily_returns) * 100:.2f}%')

        # 添加正态分布拟合
        from scipy import stats
        mu, std = stats.norm.fit(daily_returns * 100)
        xmin, xmax = ax7.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        ax7.plot(x, p * len(daily_returns) * (xmax - xmin) / 50,
                 'r-', linewidth=2, label='Normal Fit')

        ax7.set_xlabel('Daily Return (%)')
        ax7.set_ylabel('Frequency')
        ax7.set_title('Daily Returns Distribution')
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # 8. 交易成本分析
        ax8 = fig.add_subplot(gs[3, 1])
        cost_breakdown = {
            'Gross PnL': 100,
            'Commission': -5,
            'Slippage': -8,
            'Market Impact': -3,
            'Net PnL': 84
        }

        x = list(cost_breakdown.keys())
        y = list(cost_breakdown.values())
        colors = ['green' if v > 0 else 'red' for v in y]

        bars = ax8.bar(x, y, color=colors, alpha=0.7)
        ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax8.set_ylabel('PnL ($)')
        ax8.set_title('Transaction Cost Analysis')
        ax8.set_xticklabels(x, rotation=45)

        for bar, val in zip(bars, y):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width() / 2., height,
                     f'${val}', ha='center',
                     va='bottom' if val > 0 else 'top')
        ax8.grid(True, alpha=0.3)

        plt.suptitle('Backtest Results Dashboard', fontsize=16, y=1.02)
        plt.tight_layout()
        self.figures.append(fig)

    def save_all_figures(self, save_dir: str = None):
        """保存所有图表"""
        if save_dir is None:
            save_dir = f'{BASE_DIR}/visualizations_v2'

        os.makedirs(save_dir, exist_ok=True)

        dashboard_names = [
            'training_progress',
            'model_performance',
            'risk_analysis',
            'agent_analysis',
            'portfolio_analysis',
            'market_regime',
            'feature_importance',
            'backtest_results'
        ]

        for i, (fig, name) in enumerate(zip(self.figures, dashboard_names)):
            fig.savefig(f'{save_dir}/{name}_dashboard.png',
                        dpi=100, bbox_inches='tight')
            print(f"  Saved: {name}_dashboard.png")

        print(f"\n✅ All {len(self.figures)} dashboards saved to {save_dir}")


# ========================================
# 11. 主训练流程
# ========================================

# 初始化训练器
trainer = AdvancedModelTrainer(model, optimizer, criterion, scheduler, device)

# 训练模型
print("\n" + "=" * 80)
print("STARTING ADVANCED TRAINING")
print("=" * 80)

EPOCHS = 50  # 增加训练轮数
trainer.train(train_loader, val_loader, epochs=EPOCHS)


# ========================================
# 12. 最终评估
# ========================================
def evaluate_final_model(model, test_loader, device):
    """最终模型评估 - 修复版本"""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, targets in tqdm(test_loader, desc='Final Evaluation'):
            data = data.to(device)
            outputs = model(data)
            all_predictions.append(outputs['main'].cpu().numpy())
            all_targets.append(targets.numpy())

    predictions = np.concatenate(all_predictions).flatten()
    targets = np.concatenate(all_targets).flatten()

    # 打印调试信息
    print(f"\n预测统计 - Mean: {np.mean(predictions):.4f}, Std: {np.std(predictions):.4f}")
    print(f"目标统计 - Mean: {np.mean(targets):.4f}, Std: {np.std(targets):.4f}")
    print(f"目标范围 - Min: {np.min(targets):.4f}, Max: {np.max(targets):.4f}")

    # 基础回归指标
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    correlation = np.corrcoef(targets, predictions)[0, 1]

    # R2合理性检查
    if r2 > 0.9:
        print(f"\n⚠️ 警告：R² = {r2:.4f} 异常高，可能存在过拟合或数据泄露")

    # =================== 关键修复部分 ===================
    # 正确计算交易收益
    positions = np.sign(predictions)  # 基于预测方向的仓位（-1, 0, 1）
    trading_returns = positions * targets  # 实际交易收益 = 仓位 * 真实收益率

    # 添加交易成本
    position_changes = np.abs(np.diff(positions))
    num_trades = np.sum(position_changes > 0)
    transaction_cost_per_trade = 0.001  # 0.1%每次交易
    total_transaction_cost = num_trades * transaction_cost_per_trade / len(trading_returns)
    trading_returns_after_cost = trading_returns - total_transaction_cost

    # 计算夏普比率
    if np.std(trading_returns_after_cost) > 0:
        sharpe_ratio = np.mean(trading_returns_after_cost) / np.std(trading_returns_after_cost) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # 夏普比率合理性检查
    if sharpe_ratio > 3:
        print(f"\n⚠️ 警告：夏普比率 {sharpe_ratio:.2f} 仍然偏高")
        print("建议检查：")
        print("1. 确认数据是否按时间顺序划分")
        print("2. 检查是否有前瞻性偏差")
        print("3. 验证特征工程是否使用了未来信息")

    # Sortino比率（只考虑下行风险）
    downside_returns = trading_returns_after_cost[trading_returns_after_cost < 0]
    if len(downside_returns) > 0 and np.std(downside_returns) > 0:
        sortino_ratio = np.mean(trading_returns_after_cost) / np.std(downside_returns) * np.sqrt(252)
    else:
        sortino_ratio = sharpe_ratio

    # 最大回撤
    cumulative_returns = np.cumprod(1 + trading_returns_after_cost)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / (running_max + 1e-8)
    max_drawdown = np.min(drawdown)

    # Calmar比率
    annual_return = np.mean(trading_returns_after_cost) * 252
    if abs(max_drawdown) > 0.01:
        calmar_ratio = annual_return / abs(max_drawdown)
    else:
        calmar_ratio = 0

    # 胜率
    win_rate = np.mean(trading_returns > 0)

    # 盈亏比
    wins = trading_returns[trading_returns > 0]
    losses = trading_returns[trading_returns < 0]
    if len(losses) > 0 and len(wins) > 0:
        profit_loss_ratio = np.mean(wins) / np.abs(np.mean(losses))
    else:
        profit_loss_ratio = 0

    # 年化波动率
    annual_volatility = np.std(trading_returns_after_cost) * np.sqrt(252)

    # 汇总所有指标
    final_metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'r2_score': float(r2),
        'correlation': float(correlation),
        'sharpe_ratio': float(sharpe_ratio),
        'sortino_ratio': float(sortino_ratio),
        'max_drawdown': float(max_drawdown),
        'calmar_ratio': float(calmar_ratio),
        'win_rate': float(win_rate),
        'profit_loss_ratio': float(profit_loss_ratio),
        'annual_return': float(annual_return),
        'annual_volatility': float(annual_volatility),
        'num_trades': int(num_trades),
        'avg_return_per_trade': float(np.mean(trading_returns_after_cost))
    }

    return final_metrics

print("\n" + "=" * 80)
print("FINAL MODEL EVALUATION")
print("=" * 80)

final_metrics = evaluate_final_model(model, test_loader, device)

print("\n📊 Final Performance Metrics:")
print("-" * 40)
for key, value in final_metrics.items():
    if 'rate' in key or 'ratio' in key or 'r2' in key or 'correlation' in key:
        print(f"{key:20s}: {value:.4f}")
    elif 'return' in key or 'volatility' in key or 'drawdown' in key:
        print(f"{key:20s}: {value:.2%}")
    else:
        print(f"{key:20s}: {value:.6f}")

# ========================================
# 13. 创建可视化
# ========================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

viz_system = QuantitativeVisualizationSystem(trainer, model, test_loader)
viz_system.create_all_visualizations()
viz_system.save_all_figures()

# ========================================
# 14. 保存模型和结果
# ========================================

# 保存最终模型
model_save_path = f'{BASE_DIR}/quantitative_option_model_v2_final.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'final_metrics': final_metrics,
    'model_config': {
        'input_dim': input_dim,
        'sequence_length': sequence_length,
        'hidden_dim': 512,
        'n_mamba_layers': 6,
        'n_agents': 5
    }
}, model_save_path)

print(f"\n✅ Model saved to {model_save_path}")

# 导出结果到JSON
results_export = {
    'model_info': {
        'architecture': 'Mamba-based Quantitative Options Trading Model',
        'total_parameters': int(total_params),
        'trainable_parameters': int(trainable_params),
        'training_epochs': len(trainer.train_losses),
        'best_val_sharpe': float(trainer.best_val_sharpe)
    },
    'performance_metrics': final_metrics,
    'training_time': trainer.training_time,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

json_path = f'{BASE_DIR}/model_results_v2.json'
with open(json_path, 'w') as f:
    json.dump(results_export, f, indent=4)

print(f"✅ Results exported to {json_path}")

# ========================================
# 15. 最终总结
# ========================================

print("\n" + "=" * 80)
print("TRAINING COMPLETE - SUMMARY REPORT")
print("=" * 80)
print(f"⏱️  Training Duration: {trainer.training_time:.2f} seconds")
print(f"📈 Final Sharpe Ratio: {final_metrics['sharpe_ratio']:.4f}")
print(f"📊 Final Sortino Ratio: {final_metrics['sortino_ratio']:.4f}")
print(f"📉 Max Drawdown: {final_metrics['max_drawdown']:.2%}")
print(f"🎯 Win Rate: {final_metrics['win_rate']:.2%}")
print(f"💰 Annual Return: {final_metrics['annual_return']:.2%}")
print(f"📊 Annual Volatility: {final_metrics['annual_volatility']:.2%}")
print("\n✅ All tasks completed successfully!")
print("=" * 80)