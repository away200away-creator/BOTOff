class Config:
    # ==================== СТАТУС ГОТОВНОСТИ СИСТЕМЫ ====================
    ALLOW_TRADING_WITHOUT_MODEL = True  

    # ==================== СЕТЕВЫЕ НАСТРОЙКИ И API BYBIT ====================
    MAX_RETRIES = 3                      
    REQUEST_TIMEOUT = (5, 10)            
    RATE_LIMIT_PAUSE = 0.04              

    # ==================== ОСНОВНЫЕ НАСТРОЙКИ ПОРТФЕЛЯ ====================
    MAX_POSITIONS = 10                   
    MAX_DIRECTIONAL_POSITIONS = 10       
    INITIAL_BALANCE = 100.0              
    RISK_PER_TRADE = 6                 
    LEVERAGE = 3                         
    
    MAX_CONTRARIAN_POSITIONS = 5         
    MAX_BREAKOUT_POSITIONS = 5           

    # ==================== КАРАНТИНЫ И ТАЙМ-АУТЫ ====================
    COOLDOWN_MINUTES = 45                
    STOP_COOLDOWN_MINUTES = 180          
    COOLDOWNS_FILE = 'cooldowns.json'
    POSITION_UPDATE_SEC = 0.25           
    SCAN_INTERVAL_SEC = 15               
    NO_TRADE_MINUTE_START = 55           

    # ==================== СТРАТЕГИЯ 1: CONTRARIAN (MEAN REVERSION) ====================
    # Сбалансированный скоринг для регулярного интрадея
    MIN_SCORE = 38                       # Оптимальный порог (было 50)
    LIQ_Z_THRESHOLD = 0.8                # Ловим локальные сквизы розницы (было 1.5)
    OI_DROP_THRESHOLD = -0.15            # Мягкий порог сброса позиций (было -0.2)
    REQUIRE_LIQ_AND_OI = False           # Вход по CVD дельте и Price Action даже при нейтральном OI
    
    TP1_ROE = 3.0                        
    STOP_ROE = -12.0                     
    TP1_CLOSE_PCT = 50                   

    # ==================== СТРАТЕГИЯ 2: BREAKOUT (DONCHIAN EXPANSION) ====================
    BREAKOUT_LOOKBACK = 20               
    BREAKOUT_VOL_SPIKE_THRESH = 1.4      # Реалистичный всплеск объема для альткоинов (было 2.0)
    BREAKOUT_OI_GROWTH_THRESH = 0.4      # Реалистичный приток открытого интереса (было 1.0)
    BREAKOUT_MIN_SCORE = 38              # Оптимальный порог скоринга (было 50)
    
    BREAKOUT_TP1_ROE = 4.0               
    BREAKOUT_STOP_ROE = -10.0            

    # ==================== РИСК-ФИЛЬТРЫ И BTC GUARD ====================
    MAX_24H_CHANGE_FOR_SHORT = 25.0      
    BTC_FILTER = 'hard'                  
    BTC_STRONG_MOVE = 1.2                

    # ==================== МАШИННОЕ ОБУЧЕНИЕ (CATBOOST) ====================
    # Реалистичный калиброванный порог вероятности успеха для финансовых рядов
    ML_MIN_CONFIDENCE = 0.54             # Рабочий диапазон бустинга (было 0.60)

    # ==================== ПАРАМЕТРЫ РЫНКА И ИНДИКАТОРОВ ====================
    KLINE_INTERVAL = '5'                 
    MIN_VOLUME_MLN = 10.0                # Оборот от $10M дает более широкий шортлист (было 15.0)
    EMA_FAST = 20                        
    EMA_SLOW = 50                        
    RSI_PERIOD = 14                      
    ATR_PERIOD = 14                      
    LIQ_WINDOW = 15                      
    ORDER_FLOW_MAX_SUBS = 70             

    FUNDING_LONG_THRESH = 0.01           
    FUNDING_SHORT_THRESH = -0.01         
    LS_LONG_THRESH = 1.4                 
    LS_SHORT_THRESH = 0.7                
    PIN_BAR_BODY_RATIO = 0.33            
    ENABLE_ORDERBOOK_IMBALANCE = True    
    OB_IMBALANCE_THRESHOLD = 0.12        
    OB_DEPTH_PCT = 0.8                   
    RSI_OVERSOLD = 38                    
    RSI_OVERBOUGHT = 62                  

    # ==================== СОПРОВОЖДЕНИЕ, БЕЗУБЫТОК И ТРЕЙЛИНГ ====================
    BE_TRIGGER_ROE = 0.5                 
    SOFT_BE_MINUTES = 30                 
    SOFT_BE_TRIGGER_ROE = 1.0            
    SOFT_BE_LEVEL_ROE = 0.5              
    TIME_STOP_MINUTES = 45               
    TIME_STOP_MIN_ROE = 1.0              

    TRAILING_DISTANCE = 2.0              
    TRAILING_ATR_MULT = 1.0              
    TRAILING_DISTANCE_MIN = 1.5          
    TRAILING_DISTANCE_MAX = 4.0          

    # ==================== ДИНАМИЧЕСКИЙ САЙЗИНГ ПО ВОЛАТИЛЬНОСТИ ====================
    DYNAMIC_VAR_SIZING = True            
    BASE_ATR_PCT = 2.0                   
    MIN_MARGIN = 1.00                    
    MAX_MARGIN = 2.40                    

    # ==================== ТРЕЙДИНГОВОЕ ИСПОЛНЕНИЕ И КОМИССИИ BYBIT ====================
    USE_MAKER_POST_ONLY = True           
    MAKER_FEE_RATE = 0.00020             
    TAKER_FEE_RATE = 0.00055             
    SLIPPAGE_PCT = 0.08                  
    MAX_SPREAD_PCT = 0.08