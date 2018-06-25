import pandas as pd

def get_features(filename):
    '''
    Add features "Daily Returns, Moving Avg, Lagged Daily Returns, 5 Minute Returns Change"
    for each company and save the prepared dataset to file data_stocks_prepared.csv
    :param df: pandas dataframe with cleaned dataset
    :return: data_stocks_prepared.csv
    '''

    fname = filename.split(".")[0]

    # cleaning
    df = pd.read_csv(filename)

    # only retain stock prices for top 5 companies, drop remaining 495
    drop_features = ["DATE", "NASDAQ.AAL", "NASDAQ.ADBE", "NASDAQ.ADI", "NASDAQ.ADP", "NASDAQ.ADSK", "NASDAQ.AKAM",
                     "NASDAQ.ALXN", "NASDAQ.AMAT", "NASDAQ.AMD", "NASDAQ.AMGN", "NASDAQ.ATVI", "NASDAQ.AVGO",
                     "NASDAQ.BBBY", "NASDAQ.BIIB", "NASDAQ.CA", "NASDAQ.CBOE", "NASDAQ.CELG", "NASDAQ.CERN",
                     "NASDAQ.CHRW", "NASDAQ.CHTR", "NASDAQ.CINF", "NASDAQ.CMCSA", "NASDAQ.CME", "NASDAQ.COST",
                     "NASDAQ.CSCO", "NASDAQ.CSX", "NASDAQ.CTAS", "NASDAQ.CTSH", "NASDAQ.CTXS", "NASDAQ.DISCA",
                     "NASDAQ.DISCK", "NASDAQ.DISH", "NASDAQ.DLTR", "NASDAQ.EA", "NASDAQ.EBAY", "NASDAQ.EQIX",
                     "NASDAQ.ESRX", "NASDAQ.ETFC", "NASDAQ.EXPD", "NASDAQ.EXPE", "NASDAQ.FAST", "NASDAQ.FFIV",
                     "NASDAQ.FISV", "NASDAQ.FITB", "NASDAQ.FLIR", "NASDAQ.FOX", "NASDAQ.FOXA", "NASDAQ.GILD",
                     "NASDAQ.GOOGL", "NASDAQ.GRMN", "NASDAQ.GT", "NASDAQ.HAS", "NASDAQ.HBAN", "NASDAQ.HOLX",
                     "NASDAQ.HSIC", "NASDAQ.IDXX", "NASDAQ.ILMN", "NASDAQ.INCY", "NASDAQ.INFO", "NASDAQ.INTC",
                     "NASDAQ.INTU", "NASDAQ.ISRG", "NASDAQ.JBHT", "NASDAQ.KHC", "NASDAQ.KLAC", "NASDAQ.LKQ",
                     "NASDAQ.LRCX", "NASDAQ.MAR", "NASDAQ.MAT", "NASDAQ.MCHP", "NASDAQ.MDLZ", "NASDAQ.MNST",
                     "NASDAQ.MU", "NASDAQ.MYL", "NASDAQ.NAVI", "NASDAQ.NDAQ", "NASDAQ.NFLX", "NASDAQ.NTAP",
                     "NASDAQ.NTRS", "NASDAQ.NVDA", "NASDAQ.NWS", "NASDAQ.NWSA", "NASDAQ.ORLY", "NASDAQ.PAYX",
                     "NASDAQ.PBCT", "NASDAQ.PCAR", "NASDAQ.PCLN", "NASDAQ.PDCO", "NASDAQ.PYPL", "NASDAQ.QCOM",
                     "NASDAQ.QRVO", "NASDAQ.REGN", "NASDAQ.ROST", "NASDAQ.SBUX", "NASDAQ.SNI", "NASDAQ.SNPS",
                     "NASDAQ.SPLS", "NASDAQ.SRCL", "NASDAQ.STX", "NASDAQ.SWKS", "NASDAQ.SYMC", "NASDAQ.TRIP",
                     "NASDAQ.TROW", "NASDAQ.TSCO", "NASDAQ.TXN", "NASDAQ.ULTA", "NASDAQ.VIAB", "NASDAQ.VRSK",
                     "NASDAQ.VRSN", "NASDAQ.VRTX", "NASDAQ.WBA", "NASDAQ.WDC", "NASDAQ.WFM", "NASDAQ.WLTW",
                     "NASDAQ.WYNN", "NASDAQ.XLNX", "NASDAQ.XRAY", "NASDAQ.ZION", "NYSE.A", "NYSE.AAP", "NYSE.ABBV",
                     "NYSE.ABC", "NYSE.ABT", "NYSE.ACN", "NYSE.ADM", "NYSE.ADS", "NYSE.AEE", "NYSE.AEP", "NYSE.AES",
                     "NYSE.AET", "NYSE.AFL", "NYSE.AGN", "NYSE.AIG", "NYSE.AIV", "NYSE.AIZ", "NYSE.AJG", "NYSE.ALB",
                     "NYSE.ALK", "NYSE.ALL", "NYSE.ALLE", "NYSE.AME", "NYSE.AMG", "NYSE.AMP", "NYSE.AMT", "NYSE.AN",
                     "NYSE.ANTM", "NYSE.AON", "NYSE.APA", "NYSE.APC", "NYSE.APD", "NYSE.APH", "NYSE.ARE", "NYSE.ARNC",
                     "NYSE.AVB", "NYSE.AVY", "NYSE.AWK", "NYSE.AXP", "NYSE.AYI", "NYSE.AZO", "NYSE.BA", "NYSE.BAC",
                     "NYSE.BAX", "NYSE.BBT", "NYSE.BBY", "NYSE.BCR", "NYSE.BDX", "NYSE.BEN", "NYSE.BF.B", "NYSE.BHI",
                     "NYSE.BK", "NYSE.BLK", "NYSE.BLL", "NYSE.BMY", "NYSE.BRK.B", "NYSE.BSX", "NYSE.BWA", "NYSE.BXP",
                     "NYSE.C", "NYSE.CAG", "NYSE.CAH", "NYSE.CAT", "NYSE.CB", "NYSE.CBG", "NYSE.CBS", "NYSE.CCI",
                     "NYSE.CCL", "NYSE.CF", "NYSE.CFG", "NYSE.CHD", "NYSE.CHK", "NYSE.CI", "NYSE.CL", "NYSE.CLX",
                     "NYSE.CMA", "NYSE.CMG", "NYSE.CMI", "NYSE.CMS", "NYSE.CNC", "NYSE.CNP", "NYSE.COF", "NYSE.COG",
                     "NYSE.COH", "NYSE.COL", "NYSE.COO", "NYSE.COP", "NYSE.COTY", "NYSE.CPB", "NYSE.CRM", "NYSE.CSRA",
                     "NYSE.CTL", "NYSE.CVS", "NYSE.CVX", "NYSE.CXO", "NYSE.D", "NYSE.DAL", "NYSE.DD", "NYSE.DE",
                     "NYSE.DFS", "NYSE.DG", "NYSE.DGX", "NYSE.DHI", "NYSE.DHR", "NYSE.DIS", "NYSE.DLPH", "NYSE.DLR",
                     "NYSE.DOV", "NYSE.DOW", "NYSE.DPS", "NYSE.DRI", "NYSE.DTE", "NYSE.DUK", "NYSE.DVA", "NYSE.DVN",
                     "NYSE.DXC", "NYSE.ECL", "NYSE.ED", "NYSE.EFX", "NYSE.EIX", "NYSE.EL", "NYSE.EMN", "NYSE.EMR",
                     "NYSE.EOG", "NYSE.EQR", "NYSE.EQT", "NYSE.ES", "NYSE.ESS", "NYSE.ETN", "NYSE.ETR", "NYSE.EVHC",
                     "NYSE.EW", "NYSE.EXC", "NYSE.EXR", "NYSE.F", "NYSE.FBHS", "NYSE.FCX", "NYSE.FDX", "NYSE.FE",
                     "NYSE.FIS", "NYSE.FL", "NYSE.FLR", "NYSE.FLS", "NYSE.FMC", "NYSE.FRT", "NYSE.FTI", "NYSE.FTV",
                     "NYSE.GD", "NYSE.GE", "NYSE.GGP", "NYSE.GIS", "NYSE.GLW", "NYSE.GM", "NYSE.GPC", "NYSE.GPN",
                     "NYSE.GPS", "NYSE.GS", "NYSE.GWW", "NYSE.HAL", "NYSE.HBI", "NYSE.HCA", "NYSE.HCN", "NYSE.HCP",
                     "NYSE.HD", "NYSE.HES", "NYSE.HIG", "NYSE.HOG", "NYSE.HON", "NYSE.HP", "NYSE.HPE", "NYSE.HPQ",
                     "NYSE.HRB", "NYSE.HRL", "NYSE.HRS", "NYSE.HST", "NYSE.HSY", "NYSE.HUM", "NYSE.IBM", "NYSE.ICE",
                     "NYSE.IFF", "NYSE.IP", "NYSE.IPG", "NYSE.IR", "NYSE.IRM", "NYSE.IT", "NYSE.ITW", "NYSE.IVZ",
                     "NYSE.JCI", "NYSE.JEC", "NYSE.JNJ", "NYSE.JNPR", "NYSE.JPM", "NYSE.JWN", "NYSE.K", "NYSE.KEY",
                     "NYSE.KIM", "NYSE.KMB", "NYSE.KMI", "NYSE.KMX", "NYSE.KO", "NYSE.KORS", "NYSE.KR", "NYSE.KSS",
                     "NYSE.KSU", "NYSE.L", "NYSE.LB", "NYSE.LEG", "NYSE.LEN", "NYSE.LH", "NYSE.LLL", "NYSE.LLY",
                     "NYSE.LMT", "NYSE.LNC", "NYSE.LNT", "NYSE.LOW", "NYSE.LUK", "NYSE.LUV", "NYSE.LVLT", "NYSE.LYB",
                     "NYSE.M", "NYSE.MA", "NYSE.MAA", "NYSE.MAC", "NYSE.MAS", "NYSE.MCD", "NYSE.MCK", "NYSE.MCO",
                     "NYSE.MDT", "NYSE.MET", "NYSE.MHK", "NYSE.MKC", "NYSE.MLM", "NYSE.MMC", "NYSE.MMM", "NYSE.MNK",
                     "NYSE.MO", "NYSE.MON", "NYSE.MOS", "NYSE.MPC", "NYSE.MRK", "NYSE.MRO", "NYSE.MS", "NYSE.MSI",
                     "NYSE.MTB", "NYSE.MUR", "NYSE.NBL", "NYSE.NEE", "NYSE.NEM", "NYSE.NFX", "NYSE.NI", "NYSE.NKE",
                     "NYSE.NLSN", "NYSE.NOC", "NYSE.NOV", "NYSE.NRG", "NYSE.NSC", "NYSE.NUE", "NYSE.NWL", "NYSE.O",
                     "NYSE.OKE", "NYSE.OMC", "NYSE.ORCL", "NYSE.OXY", "NYSE.PCG", "NYSE.PEG", "NYSE.PEP", "NYSE.PFE",
                     "NYSE.PFG", "NYSE.PG", "NYSE.PGR", "NYSE.PH", "NYSE.PHM", "NYSE.PKI", "NYSE.PLD", "NYSE.PM",
                     "NYSE.PNC", "NYSE.PNR", "NYSE.PNW", "NYSE.PPG", "NYSE.PPL", "NYSE.PRGO", "NYSE.PRU", "NYSE.PSA",
                     "NYSE.PSX", "NYSE.PVH", "NYSE.PWR", "NYSE.PX", "NYSE.PXD", "NYSE.RAI", "NYSE.RCL", "NYSE.REG",
                     "NYSE.RF", "NYSE.RHI", "NYSE.RHT", "NYSE.RIG", "NYSE.RJF", "NYSE.RL", "NYSE.ROK", "NYSE.ROP",
                     "NYSE.RRC", "NYSE.RSG", "NYSE.RTN", "NYSE.SCG", "NYSE.SCHW", "NYSE.SEE", "NYSE.SHW", "NYSE.SIG",
                     "NYSE.SJM", "NYSE.SLB", "NYSE.SLG", "NYSE.SNA", "NYSE.SO", "NYSE.SPG", "NYSE.SPGI", "NYSE.SRE",
                     "NYSE.STI", "NYSE.STT", "NYSE.STZ", "NYSE.SWK", "NYSE.SYF", "NYSE.SYK", "NYSE.SYY", "NYSE.T",
                     "NYSE.TAP", "NYSE.TDG", "NYSE.TEL", "NYSE.TGT", "NYSE.TIF", "NYSE.TJX", "NYSE.TMK", "NYSE.TMO",
                     "NYSE.TRV", "NYSE.TSN", "NYSE.TSO", "NYSE.TSS", "NYSE.TWX", "NYSE.TXT", "NYSE.UA", "NYSE.UAA",
                     "NYSE.UAL", "NYSE.UDR", "NYSE.UHS", "NYSE.UNH", "NYSE.UNM", "NYSE.UNP", "NYSE.UPS", "NYSE.URI",
                     "NYSE.USB", "NYSE.UTX", "NYSE.V", "NYSE.VAR", "NYSE.VFC", "NYSE.VLO", "NYSE.VMC", "NYSE.VNO",
                     "NYSE.VTR", "NYSE.VZ", "NYSE.WAT", "NYSE.WEC", "NYSE.WFC", "NYSE.WHR", "NYSE.WM", "NYSE.WMB",
                     "NYSE.WMT", "NYSE.WRK", "NYSE.WU", "NYSE.WY", "NYSE.WYN", "NYSE.XEC", "NYSE.XEL", "NYSE.XL",
                     "NYSE.XOM", "NYSE.XRX", "NYSE.XYL", "NYSE.YUM", "NYSE.ZBH", "NYSE.ZTS"]
    df.drop(drop_features, axis=1, inplace=True)

    # preparation
    for col_name in df.columns.values:
        df[col_name + '_M_RET'] = df[col_name].pct_change(1)
        df[col_name + '_5M_RET'] = df[col_name].pct_change(5)
        df[col_name + '_5M_AVG'] = df[col_name+'_M_RET'].rolling(window=5, center=False).mean()
        df[col_name + '_LAGGED'] = df[col_name+'_M_RET'].shift(1)

    drop_features = ['AAPL', 'AMZN', 'FB', 'GOOG', 'MSFT', 'SP500_LAGGED', 'SP500']
    df.drop(drop_features, axis=1, inplace=True)
    df = df[5:]
    df.to_csv(path_or_buf=fname + '_prepared.csv', index=False)
    # print('Columns: ', df.columns.values)

    return df