import numpy as np
import tkinter as tk
import pandas_datareader.data as web
import datetime as dt
from scipy import stats
from PIL import ImageTk, Image
from tkinter import messagebox, ttk

class Page1(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        # variables init
        self.vars = {"TTM": 1, "Strike Price": 100, "Interest Rate": 0.1,
                     "Implied Vol1": 0, "Implied Vol2": 0, "Price 1": 0,
                     "Price 2": 0, "Vol 1": 0, "Vol 2": 0, "rho_vh": 1,
                     "ticker1": None, "ticker2": None, "rho_vh_mat": None,
                     }

        # inputs frame
        input_frame = tk.Frame(self, width=350, height=500)
        input_frame.place(x=50, y=50)

        # Asset tickers
        asset1 = tk.Label(input_frame, text="Asset 1", font="Helvetica 9 bold")
        self.asset1_label = tk.Entry(input_frame, width=11)
        self.asset1_label.insert(0, "AAPL")
        asset1.place(in_=input_frame, x=0, y=0)
        self.asset1_label.place(in_=input_frame, x=100, y=0)

        asset2 = tk.Label(input_frame, text="Asset 2", font="Helvetica 9 bold")
        self.asset2_label = tk.Entry(input_frame, width=11)
        self.asset2_label.insert(0, "AMZN")
        asset2.place(in_=input_frame, x=0, y=30)
        self.asset2_label.place(in_=input_frame, x=100, y=30)

        # essential inputs
        TTM = tk.Label(input_frame, text="Time to Maturity", font="Helvetica 9 bold")
        self.TTM_label = tk.Entry(input_frame, width=11)
        self.TTM_label.insert(0, self.vars["TTM"])
        TTM.place(in_=input_frame, x=0, y=60)
        self.TTM_label.place(in_=input_frame, x=100, y=60)

        X_price = tk.Label(input_frame, text="Strike Price", font="Helvetica 9 bold")
        self.X_price_label = tk.Entry(input_frame, width=11)
        self.X_price_label.insert(0, self.vars["Strike Price"])
        X_price.place(in_=input_frame, x=0, y=90)
        self.X_price_label.place(in_=input_frame, x=100, y=90)

        int_rate = tk.Label(input_frame, text="Interest Rate", font="Helvetica 9 bold")
        self.int_rate_label = tk.Entry(input_frame, width=11)
        self.int_rate_label.insert(0, self.vars["Interest Rate"])
        int_rate.place(in_=input_frame, x=0, y=120)
        self.int_rate_label.place(in_=input_frame, x=100, y=120)

        # optional inputs
        imp_vol1 = tk.Label(input_frame, text="Implied Vol1", font="Helvetica 9 bold")
        self.imp_vol1_label = tk.Entry(input_frame, width=11)
        self.imp_vol1_label.insert(0, self.vars["Implied Vol1"])
        imp_vol1.place(in_=input_frame, x=0, y=150)
        self.imp_vol1_label.place(in_=input_frame, x=100, y=150)

        imp_vol2 = tk.Label(input_frame, text="Implied Vol2", font="Helvetica 9 bold")
        self.imp_vol2_label = tk.Entry(input_frame, width=11)
        self.imp_vol2_label.insert(0, self.vars["Implied Vol2"])
        imp_vol2.place(in_=input_frame, x=0, y=180)
        self.imp_vol2_label.place(in_=input_frame, x=100, y=180)

        # up and down arrow buttons
        img_up = Image.open("./images/up_icon.jpg").resize((5, 5))
        img_up = ImageTk.PhotoImage(img_up)
        img_down = Image.open("./images/down_icon.jpg").resize((5, 5))
        img_down = ImageTk.PhotoImage(img_down)

        # ttm up and down button
        b_ttmup = tk.Button(input_frame, image=img_up, command=self.ttm_up)
        b_ttmup.image = img_up
        b_ttmdown = tk.Button(input_frame, image=img_down, command=self.ttm_down)
        b_ttmdown.image = img_down

        # strike price up and down button
        b_xup = tk.Button(input_frame, image=img_up, command=self.b_xup)
        b_xup.image = img_up
        b_xdown = tk.Button(input_frame, image=img_down, command=self.b_xdown)
        b_xdown.image = img_down

        # interest rate up and down button
        b_intup = tk.Button(input_frame, image=img_up, command=self.b_intup)
        b_intup.image = img_up
        b_intdown = tk.Button(input_frame, image=img_down, command=self.b_intdown)
        b_intdown.image = img_down

        # vol up and down button
        b_vol1up = tk.Button(input_frame, image=img_up, command=self.b_vol1up)
        b_vol1up.image = img_up
        b_vol1down = tk.Button(input_frame, image=img_down, command=self.b_vol1down)
        b_vol1down.image = img_down

        b_vol2up = tk.Button(input_frame, image=img_up, command=self.b_vol2up)
        b_vol2up.image = img_up
        b_vol2down = tk.Button(input_frame, image=img_down, command=self.b_vol2down)
        b_vol2down.image = img_down

        # place up and down buttons
        b_ttmup.place(in_=input_frame, x=170, y=60)
        b_ttmdown.place(in_=input_frame, x=170, y=70)
        b_xup.place(in_=input_frame, x=170, y=90)
        b_xdown.place(in_=input_frame, x=170, y=100)
        b_intup.place(in_=input_frame, x=170, y=120)
        b_intdown.place(in_=input_frame, x=170, y=130)
        b_vol1up.place(in_=input_frame, x=170, y=150)
        b_vol1down.place(in_=input_frame, x=170, y=160)
        b_vol2up.place(in_=input_frame, x=170, y=180)
        b_vol2down.place(in_=input_frame, x=170, y=190)

        # price model
        b_pricing = tk.Button(input_frame, text='Start Pricing', command=self.pricing)
        b_pricing.place(in_=input_frame, x=0, y=300)

        # calculated display frame
        cal_frame = tk.Frame(self, width=200, height=500)
        cal_frame.place(x=300, y=0)
        # assets prices
        pc1 = tk.Label(cal_frame, text="Asset 1 Price:", font="Helvetica 9 bold")
        self.pc1_con = tk.Label(cal_frame, text="placeholder")
        pc1.place(in_=cal_frame, x=0, y=50)
        self.pc1_con.place(in_=cal_frame, x=90, y=50)

        pc2 = tk.Label(cal_frame, text="Asset 2 Price:", font="Helvetica 9 bold")
        self.pc2_con = tk.Label(cal_frame, text="placeholder")
        pc2.place(in_=cal_frame, x=0, y=80)
        self.pc2_con.place(in_=cal_frame, x=90, y=80)

        # calculated vol and rho
        vol1 = tk.Label(cal_frame, text="Asset 1 Vol:", font="Helvetica 9 bold")
        self.vol1_con = tk.Label(cal_frame, text="placeholder")
        vol1.place(in_=cal_frame, x=0, y=110)
        self.vol1_con.place(in_=cal_frame, x=90, y=110)

        vol2 = tk.Label(cal_frame, text="Asset 2 Vol:", font="Helvetica 9 bold")
        self.vol2_con = tk.Label(cal_frame, text="placeholder")
        vol2.place(in_=cal_frame, x=0, y=140)
        self.vol2_con.place(in_=cal_frame, x=90, y=140)

        rho_text = u"\u03c1\N{LATIN SUBSCRIPT SMALL LETTER V}\N{LATIN SUBSCRIPT SMALL LETTER h} :"
        rho_cal = tk.Label(cal_frame, text=rho_text, font="Helvetica 9 bold")
        self.rho_cal_con = tk.Label(cal_frame, text="placeholder")
        rho_cal.place(in_=cal_frame, x=0, y=170)
        self.rho_cal_con.place(in_=cal_frame, x=90, y=170)

        # output frame
        output_frame = tk.Frame(self, width=350, height=250)
        output_frame.place(x=500, y=0)

        # option price
        option_price = tk.Label(output_frame, text="Call-Min Option Price:", font="Helvetica 9 bold")
        self.option_price_con = tk.Label(output_frame, text="placeholder")
        option_price.place(in_=output_frame, x=0, y=50)
        self.option_price_con.place(in_=output_frame, x=130, y=50)

        # greeks
        dt1_out = tk.Label(output_frame, text="Delta 1:", font="Helvetica 9 bold")
        self.dt1_out_con = tk.Label(output_frame, text="placeholder")
        dt1_out.place(in_=output_frame, x=0, y=80)
        self.dt1_out_con.place(in_=output_frame, x=50, y=80)

        dt2_out = tk.Label(output_frame, text="Delta 2:", font="Helvetica 9 bold")
        self.dt2_out_con = tk.Label(output_frame, text="placeholder")
        dt2_out.place(in_=output_frame, x=0, y=110)
        self.dt2_out_con.place(in_=output_frame, x=50, y=110)

        vg1_out = tk.Label(output_frame, text="Vega 1:", font="Helvetica 9 bold")
        self.vg1_out_con = tk.Label(output_frame, text="placeholder")
        vg1_out.place(in_=output_frame, x=0, y=140)
        self.vg1_out_con.place(in_=output_frame, x=50, y=140)

        vg2_out = tk.Label(output_frame, text="Vega 2:", font="Helvetica 9 bold")
        self.vg2_out_con = tk.Label(output_frame, text="placeholder")
        vg2_out.place(in_=output_frame, x=0, y=170)
        self.vg2_out_con.place(in_=output_frame, x=50, y=170)

        rho_out = tk.Label(output_frame, text="Rho:", font="Helvetica 9 bold")
        self.rho_out_con = tk.Label(output_frame, text="placeholder")
        rho_out.place(in_=output_frame, x=160, y=80)
        self.rho_out_con.place(in_=output_frame, x=200, y=80)

        cora_out = tk.Label(output_frame, text="Cora:", font="Helvetica 9 bold")
        self.cora_out_con = tk.Label(output_frame, text="placeholder")
        cora_out.place(in_=output_frame, x=160, y=110)
        self.cora_out_con.place(in_=output_frame, x=200, y=110)

        gora_out = tk.Label(output_frame, text="Gora:", font="Helvetica 9 bold")
        self.gora_out_con = tk.Label(output_frame, text="placeholder")
        gora_out.place(in_=output_frame, x=160, y=140)
        self.gora_out_con.place(in_=output_frame, x=200, y=140)

        # pricing matrix frame
        matrix_frame = tk.Frame(self, width=350, height=250)
        matrix_frame.place(x=500, y=250)
        mat_call = tk.Label(matrix_frame, text="Call", font="Helvetica 9 bold")
        mat_put = tk.Label(matrix_frame, text="Put", font="Helvetica 9 bold")
        mat_max = tk.Label(matrix_frame, text="Max", font="Helvetica 9 bold")
        mat_min = tk.Label(matrix_frame, text="Min", font="Helvetica 9 bold")
        self.mat_call_max = tk.Label(matrix_frame, text="placeholder")
        self.mat_call_min = tk.Label(matrix_frame, text="placeholder")
        self.mat_put_max = tk.Label(matrix_frame, text="placeholder")
        self.mat_put_min = tk.Label(matrix_frame, text="placeholder")
        mat_call.place(in_=matrix_frame, x=0, y=50)
        mat_put.place(in_=matrix_frame, x=0, y=100)
        mat_max.place(in_=matrix_frame, x=50, y=0)
        mat_min.place(in_=matrix_frame, x=150, y=0)
        self.mat_call_max.place(in_=matrix_frame, x=50, y=50)
        self.mat_call_min.place(in_=matrix_frame, x=150, y=50)
        self.mat_put_max.place(in_=matrix_frame, x=50, y=100)
        self.mat_put_min.place(in_=matrix_frame, x=150, y=100)

    # utility functions
    def pricing(self):
        ticker1 = self.asset1_label.get().strip().upper()
        ticker2 = self.asset2_label.get().strip().upper()
        if ticker1 != self.vars["ticker1"] or ticker2 != self.vars["ticker2"]:
            self.vars["ticker1"] = ticker1
            self.vars["ticker2"] = ticker2
            self.get_data_cal(ticker1, ticker2)

        # update inputs
        self.vars["Strike Price"] = float(self.X_price_label.get())
        self.vars["Interest Rate"] = float(self.int_rate_label.get())
        self.vars["TTM"] = float(self.TTM_label.get())
        self.vars["Implied Vol1"] = float(self.imp_vol1_label.get())
        self.vars["Implied Vol2"] = float(self.imp_vol2_label.get())

        # calculate and update on dashboard
        self.params_cal()

    def get_data_cal(self, ticker1, ticker2):
        tickers = [ticker1, ticker2]
        end = dt.datetime.now()
        start = end - dt.timedelta(days=365)
        df_prices = web.DataReader(name=tickers, data_source="yahoo", start=start, end=end)["Adj Close"]
        returns = df_prices.pct_change().dropna()
        rho_vh_mat = returns.corr().to_numpy()
        rho_vh = rho_vh_mat[0, 1]
        V, H = df_prices.iloc[-1]
        sigma_v, sigma_h = returns.std()
        sigma_v, sigma_h = sigma_v * np.sqrt(252), sigma_h * np.sqrt(252)
        # sigma
        sigma = np.sqrt(sigma_h ** 2 + sigma_v ** 2 - 2 * rho_vh * sigma_h * sigma_v)
        self.vars["Price 1"] = V
        self.vars["Price 2"] = H
        self.vars["sigma_v"] = sigma_v
        self.vars["sigma_h"] = sigma_h
        self.vars["sigma"] = sigma
        self.vars["rho_vh"] = rho_vh
        self.vars["rho_vh_mat"] = rho_vh_mat

        # update on the dashboard
        self.pc1_con.config(text="{:.2f}".format(V))
        self.pc2_con.config(text="{:.2f}".format(H))
        self.vol1_con.config(text="{:.2f}".format(sigma_v))
        self.vol2_con.config(text="{:.2f}".format(sigma_h))
        self.rho_cal_con.config(text="{:.2f}".format(rho_vh))

    def params_cal(self):
        # take out vars
        V = self.vars["Price 1"]
        H = self.vars["Price 2"]
        sigma_v = self.vars["sigma_v"]
        sigma_h = self.vars["sigma_h"]
        F = self.vars["Strike Price"]
        R = self.vars["Interest Rate"]
        T = self.vars["TTM"]
        rho_vh = self.vars["rho_vh"]
        rho_vh_mat = self.vars["rho_vh_mat"]

        # check if using implied vols
        implied_vol1 = self.vars["Implied Vol1"]
        implied_vol2 = self.vars["Implied Vol2"]
        if implied_vol1 != 0:
            sigma_v = implied_vol1
        if implied_vol2 != 0:
            sigma_h = implied_vol2

        sigma = np.sqrt(sigma_h ** 2 + sigma_v ** 2 - 2 * rho_vh * sigma_h * sigma_v)

        # param calculations
        # gammas
        gamma1 = (np.log(H / F) + (R - 0.5 * sigma_h ** 2) * T) / (sigma_h * np.sqrt(T))
        gamma2 = (np.log(V / F) + (R - 0.5 * sigma_v ** 2) * T) / (sigma_v * np.sqrt(T))

        # alphas, betas, rhos
        t1_alpha1 = gamma1 + sigma_h * np.sqrt(T)
        t1_alpha2 = (np.log(V / H) - 0.5 * np.sqrt(T) * sigma ** 2) / (sigma * np.sqrt(T))
        t1_rho = (rho_vh * sigma_v - sigma_h) / sigma
        t1_rho_mat = np.ones((2, 2))
        t1_rho_mat[[0, 1], [1, 0]] = t1_rho

        t2_beta1 = gamma2 + sigma_v * np.sqrt(T)
        t2_beta2 = (np.log(H / V) - 0.5 * np.sqrt(T) * sigma ** 2) / (sigma * np.sqrt(T))
        t2_rho = (rho_vh * sigma_h - sigma_v) / sigma
        t2_rho_mat = np.ones((2, 2))
        t2_rho_mat[[0, 1], [1, 0]] = t2_rho

        # bivariates normals
        norm_t1 = stats.multivariate_normal([0, 0], t1_rho_mat)
        norm_t2 = stats.multivariate_normal([0, 0], t2_rho_mat)
        norm_t3 = stats.multivariate_normal([0, 0], rho_vh_mat)

        # discount factors
        df_alpha = np.exp(-0.5 * t1_alpha2 ** 2)
        df_beta = np.exp(-0.5 * t2_beta2 ** 2)

        # Greeks calculation
        # vega
        norm = stats.norm(0, 1)
        d_v = (np.log(V / H) + (R - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega_v = -H * np.exp(-R * T) * norm.pdf(d_v) * (np.sqrt(T) / sigma) * (1 - rho_vh * sigma_v / sigma_h)

        norm = stats.norm(0, 1)
        d_h = (np.log(H / V) + (R - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega_h = -V * np.exp(-R * T) * norm.pdf(d_h) * (np.sqrt(T) / sigma) * (1 - rho_vh * sigma_h / sigma_v)

        # deltas
        k = 1
        delta_h_d1 = (t1_alpha1 - t1_rho * t1_alpha2) / (np.sqrt(1 - t1_rho ** 2))
        delta_h_d2 = (t2_beta1 - t2_rho * t2_beta2) / (np.sqrt(1 - t2_rho ** 2))

        delta_h = norm_t1.cdf([t1_alpha1, t1_alpha2])
        delta_h += df_alpha * norm.cdf(delta_h_d1) / (k ** 0.5)
        delta_h -= (V / H) * df_beta * norm.cdf(delta_h_d2) / (k ** 0.5)

        delta_v_d1 = delta_h_d2
        delta_v_d2 = delta_h_d1

        delta_v = norm_t2.cdf([t2_beta1, t2_beta2])
        delta_v += df_beta * norm.cdf(delta_v_d1) / (k ** 0.5)
        delta_v -= (H / V) * df_alpha * norm.cdf(delta_v_d2) / (k ** 0.5)

        # Rho
        Rho = T * np.exp(-R * T) * norm_t3.cdf([gamma1, gamma2])

        # Cora
        cora_d1 = delta_h_d1
        cora_d2 = delta_h_d2

        cora_p = sigma_v * sigma_h * np.sqrt(T) / sigma
        cora_1 = cora_p + sigma_v * sigma_h * t1_alpha2 / (sigma ** 2)
        cora_2 = cora_p + sigma_v * sigma_h * t2_beta2 / (sigma ** 2)

        cora_t1 = H * norm.cdf(cora_d1) * df_alpha * cora_1 / (np.sqrt(2 * np.pi))
        cora_t2 = V * norm.cdf(cora_d2) * df_beta * cora_2 / (np.sqrt(2 * np.pi))
        cora = cora_t1 + cora_t2

        # Gora
        gora_m1 = -t1_alpha2 * np.sqrt(1 - t1_rho ** 2)
        gora_m1 += (t1_alpha1 - t1_rho * t1_alpha2) * t1_rho / np.sqrt(1 - t1_rho ** 2)
        gora_m1 /= 1 - t1_rho ** 2

        gora_m2 = -t2_beta2 * np.sqrt(1 - t2_rho ** 2)
        gora_m2 += (t2_beta1 - t2_rho * t2_beta2) * t2_rho / np.sqrt(1 - t2_rho ** 2)
        gora_m2 /= 1 - t2_rho ** 2

        gora_d1 = delta_h_d1
        gora_d2 = delta_h_d2

        gora_t1 = cora_t1 * norm.pdf(gora_d1) * gora_m1
        gora_t2 = cora_t2 * norm.pdf(gora_d2) * gora_m2
        gora = gora_t1 + gora_t2

        # call-max
        MX_price = self.MX(V=V, H=H, F=F, T=T, sigma_v=sigma_v, sigma_h=sigma_h, rho_vh=rho_vh, R=R)
        # call-min
        M_price = self.M(V=V, H=H, F=F, T=T, sigma_v=sigma_v, sigma_h=sigma_h, rho_vh=rho_vh, R=R)
        # put-max
        PX_price = self.PX(V=V, H=H, F=F, T=T, sigma_v=sigma_v, sigma_h=sigma_h, rho_vh=rho_vh, R=R)
        # put-min
        PM_price = self.PM(V=V, H=H, F=F, T=T, sigma_v=sigma_v, sigma_h=sigma_h, rho_vh=rho_vh, R=R)

        # update calculated vars
        # update output frame
        self.option_price_con.config(text="{:.2f}".format(M_price))
        self.dt1_out_con.config(text="{:.4f}".format(delta_v))
        self.dt2_out_con.config(text="{:.4f}".format(delta_h))
        self.vg1_out_con.config(text="{:.4f}".format(vega_v))
        self.vg2_out_con.config(text="{:.4f}".format(vega_h))
        self.rho_out_con.config(text="{:.4f}".format(Rho))
        self.cora_out_con.config(text="{:.4f}".format(cora))
        self.gora_out_con.config(text="{:.4f}".format(gora))

        # update matrix frame
        self.mat_call_max.config(text="{:.2f}".format(MX_price))
        self.mat_call_min.config(text="{:.2f}".format(M_price))
        self.mat_put_max.config(text="{:.2f}".format(PX_price))
        self.mat_put_min.config(text="{:.2f}".format(PM_price))

    # pricing functions
    # European call option
    def C(self, S, F, T, sigma, R):
        if F == 0: return S
        d1 = (np.log(S / F) + (R + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        norm = stats.norm(0, 1)
        C = S * norm.cdf(d1) - F * np.exp(-R * T) * norm.cdf(d2)
        return C

    # call options on min of two
    def M(self, V, H, F, T, sigma_v, sigma_h, rho_vh, R):
        non_zero = 1
        if F == 0:
            non_zero = 0
            F = 1
        # call options on min of two risky assets
        sigma = np.sqrt(sigma_h ** 2 + sigma_v ** 2 - 2 * rho_vh * sigma_h * sigma_v)

        # gammas
        gamma1 = (np.log(H / F) + (R - 0.5 * sigma_h ** 2) * T) / (sigma_h * np.sqrt(T))
        gamma2 = (np.log(V / F) + (R - 0.5 * sigma_v ** 2) * T) / (sigma_v * np.sqrt(T))

        # alphas
        t1_alpha1 = gamma1 + sigma_h * np.sqrt(T)
        t1_alpha2 = (np.log(V / H) - 0.5 * np.sqrt(T) * sigma ** 2) / (sigma * np.sqrt(T))
        t1_rho = (rho_vh * sigma_v - sigma_h) / sigma
        t1_rho_mat = np.ones((2, 2))
        t1_rho_mat[[0, 1], [1, 0]] = t1_rho
        # betas
        t2_beta1 = gamma2 + sigma_v * np.sqrt(T)
        t2_beta2 = (np.log(H / V) - 0.5 * np.sqrt(T) * sigma ** 2) / (sigma * np.sqrt(T))
        t2_rho = (rho_vh * sigma_h - sigma_v) / sigma
        t2_rho_mat = np.ones((2, 2))
        t2_rho_mat[[0, 1], [1, 0]] = t2_rho

        # bivariate normals
        rho_vh_mat = self.vars["rho_vh_mat"]
        norm_t1 = stats.multivariate_normal([0, 0], t1_rho_mat)
        norm_t2 = stats.multivariate_normal([0, 0], t2_rho_mat)
        norm_t3 = stats.multivariate_normal([0, 0], rho_vh_mat)

        t1 = H * norm_t1.cdf([t1_alpha1, t1_alpha2])
        t2 = V * norm_t2.cdf([t2_beta1, t2_beta2])
        t3 = F * np.exp(-R * T) * norm_t3.cdf([gamma1, gamma2])

        M = t1 + t2 - t3
        return M if non_zero == 1 else V - M

    # put option on min of two
    def PM(self, V, H, F, T, sigma_v, sigma_h, rho_vh, R):
        m_0 = self.M(V=V, H=H, F=0, T=T, sigma_v=sigma_v, sigma_h=sigma_h, rho_vh=rho_vh, R=R)
        m = self.M(V=V, H=H, F=F, T=T, sigma_v=sigma_v, sigma_h=sigma_h, rho_vh=rho_vh, R=R)
        return np.exp(-R * T) * F - m_0 + m

    # call option on the max of two risky assets
    def MX(self, V, H, F, T, sigma_v, sigma_h, rho_vh, R):
        c_v = self.C(S=V, F=F, T=T, sigma=sigma_v, R=R)
        c_h = self.C(S=H, F=F, T=T, sigma=sigma_h, R=R)
        m = self.M(V=V, H=H, F=F, T=T, sigma_v=sigma_v, sigma_h=sigma_h, rho_vh=rho_vh, R=R)
        return c_v + c_h - m

    # put option on max of two
    def PX(self, V, H, F, T, sigma_v, sigma_h, rho_vh, R):
        mx1 = self.MX(V=V, H=H, F=0, T=T, sigma_v=sigma_v, sigma_h=sigma_h, rho_vh=rho_vh, R=R)
        mx2 = self.MX(V=V, H=H, F=F, T=T, sigma_v=sigma_v, sigma_h=sigma_h, rho_vh=rho_vh, R=R)
        return np.exp(-R * T) * F - mx1 + mx2

    # command functions
    # time to maturity control button
    def ttm_down(self):
        if self.vars['TTM'] - 0.1 < 0:
            messagebox.showwarning(title="Value outflow", message="Time to maturity should be positive")
        else:
            self.vars['TTM'] -= 0.1
            self.TTM_label.delete(0, "end")
            self.TTM_label.insert(0, "{:.1f}".format(self.vars['TTM']))

    def ttm_up(self):
        self.vars['TTM'] += 0.1
        self.TTM_label.delete(0, "end")
        self.TTM_label.insert(0, "{:.1f}".format(self.vars['TTM']))

    # strike price control button
    def b_xdown(self):
        if self.vars["Strike Price"] - 1 < 0:
            messagebox.showwarning(title="Value outflow", message="Strike Price should be positive")
        else:
            self.vars["Strike Price"] -= 1
            self.X_price_label.delete(0, "end")
            self.X_price_label.insert(0, "{:.1f}".format(self.vars["Strike Price"]))

    def b_xup(self):
        self.vars["Strike Price"] += 1
        self.X_price_label.delete(0, "end")
        self.X_price_label.insert(0, "{:.1f}".format(self.vars["Strike Price"]))

    # interest rate control buttons
    def b_intdown(self):
        if self.vars["Interest Rate"] - 0.01 < 0:
            messagebox.showwarning(title="Value outflow", message="Interest Rate should be positive")
        else:
            self.vars["Interest Rate"] -= 0.01
            self.int_rate_label.delete(0, "end")
            self.int_rate_label.insert(0, "{:.2f}".format(self.vars["Interest Rate"]))

    def b_intup(self):
        self.vars["Interest Rate"] += 0.01
        self.int_rate_label.delete(0, "end")
        self.int_rate_label.insert(0, "{:.2f}".format(self.vars["Interest Rate"]))

    # implied vol control buttons
    def b_vol1down(self):
        if self.vars["Implied Vol1"] - 0.01 < 0:
            messagebox.showwarning(title="Value outflow", message="Implied volatility should be positive")
        else:
            self.vars["Implied Vol1"] -= 0.01
            self.imp_vol1_label.delete(0, "end")
            self.imp_vol1_label.insert(0, "{:.2f}".format(self.vars["Implied Vol1"]))

    def b_vol1up(self):
        if self.vars["Implied Vol1"] + 0.01 > 1:
            messagebox.showwarning(title="Value outflow", message="Implied volatility cannot be greater than 1")
        else:
            self.vars["Implied Vol1"] += 0.01
            self.imp_vol1_label.delete(0, "end")
            self.imp_vol1_label.insert(0, "{:.2f}".format(self.vars["Implied Vol1"]))

    def b_vol2down(self):
        if self.vars["Implied Vol2"] - 0.01 < 0:
            messagebox.showwarning(title="Value outflow", message="Implied volatility should be positive")
        else:
            self.vars["Implied Vol2"] -= 0.01
            self.imp_vol2_label.delete(0, "end")
            self.imp_vol2_label.insert(0, "{:.2f}".format(self.vars["Implied Vol2"]))

    def b_vol2up(self):
        if self.vars["Implied Vol2"] + 0.01 > 1:
            messagebox.showwarning(title="Value outflow", message="Implied volatility cannot be greater than 1")
        else:
            self.vars["Implied Vol2"] += 0.01
            self.imp_vol2_label.delete(0, "end")
            self.imp_vol2_label.insert(0, "{:.2f}".format(self.vars["Implied Vol2"]))


class Page2(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        # init graphs
        c_min = Image.open("./images/call_min_of_two.png").resize((500, 500))
        c_min = ImageTk.PhotoImage(c_min)

        c_max = Image.open("./images/call_max_of_two.png").resize((500, 500))
        c_max = ImageTk.PhotoImage(c_max)

        p_min = Image.open("./images/put_min_of_two.png").resize((500, 500))
        p_min = ImageTk.PhotoImage(p_min)

        p_max = Image.open("./images/put_max_of_two.png").resize((500, 500))
        p_max = ImageTk.PhotoImage(p_max)

        call = Image.open("./images/Call_option_payoff.png").resize((500, 500))
        call = ImageTk.PhotoImage(call)

        put = Image.open("./images/Put_option_payoff.png").resize((500, 500))
        put = ImageTk.PhotoImage(put)

        self.graph_dic = {"Call option on min of two risky assets": c_min,
                          "Call option on max of two risky assets": c_max,
                          "Put option on min of two risky assets": p_min,
                          "Put option on max of two risky assets": p_max,
                          "Call Option": call,
                          "Put Option": put}
        self.variable = tk.StringVar(self)
        self.variable.set("Call option")
        w = tk.OptionMenu(self, self.variable, *self.graph_dic.keys(), command=self.change_graph)
        w.pack()
        self.display = tk.Label(self, image=call)
        self.display.image = c_min
        self.display.pack()

    def change_graph(self, curr):
        self.display.config(image=self.graph_dic[curr])
        self.display.image = self.graph_dic[curr]


class Page3(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        op_p = Image.open("./images/reference/option_price.JPG").resize((500, 300))
        op_p = ImageTk.PhotoImage(op_p)

        call_max = Image.open("./images/reference/call_max_parity.JPG").resize((500, 200))
        call_max = ImageTk.PhotoImage(call_max)

        put_max = Image.open("./images/reference/put_max_parity.JPG").resize((500, 200))
        put_max = ImageTk.PhotoImage(put_max)

        put_min = Image.open("./images/reference/put_min_parity.JPG").resize((500, 150))
        put_min = ImageTk.PhotoImage(put_min)

        cora = Image.open("./images/reference/cora.JPG").resize((500, 250))
        cora = ImageTk.PhotoImage(cora)

        vega = Image.open("./images/reference/vega.JPG").resize((500, 300))
        vega = ImageTk.PhotoImage(vega)

        appendix = Image.open("./images/reference/appendix.JPG").resize((500, 350))
        appendix = ImageTk.PhotoImage(appendix)

        self.rf_list = ["Option Pricing", "Call Max, Parity", "Put Min, Parity",
                   "Put Max, Parity", "Cora", "Vega", "Appendix"]
        self.rf_dic = {"Option Pricing": op_p, "Call Max, Parity": call_max, "Put Min, Parity": put_max,
                  "Put Max, Parity": put_min, "Cora": cora, "Vega": vega, "Appendix": appendix}

        self.refer_dis = tk.Label(self, image=op_p)
        self.refer_dis.image = op_p
        self.refer_dis.place(in_=self, x=200, y=50)

        rf_var = tk.StringVar(value=self.rf_list)
        self.listbox = tk.Listbox(self, listvariable=rf_var, height=6, selectmode="extended")
        self.listbox.place(in_=self, x=20, y=50)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.listbox.yview)
        scrollbar.place(in_=self, x=150, y=50)

        self.listbox.bind("<<ListboxSelect>>", self.items_selected)

    def items_selected(self, event):
        idx = self.listbox.curselection()[0]
        img = self.rf_dic[self.rf_list[idx]]
        self.refer_dis.config(image=img)
        self.refer_dis.image = img


class MainView(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        p1 = Page1(self)
        p2 = Page2(self)
        p3 = Page3(self)

        buttonframe = tk.Frame(self)
        container = tk.Frame(self)
        buttonframe.pack(side="top", fill="x", expand=False)
        container.pack(side="top", fill="both", expand=True)

        p1.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        p2.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        p3.place(in_=container, x=0, y=0, relwidth=1, relheight=1)

        b1 = tk.Button(buttonframe, text="Pricing Model", command=p1.lift)
        b2 = tk.Button(buttonframe, text="Graphs", command=p2.lift)
        b3 = tk.Button(buttonframe, text="Formulas", command=p3.lift)

        b1.pack(side='left')
        b2.pack(side='left')
        b3.pack(side='left')

        p1.lift()


if __name__ == '__main__':
    root = tk.Tk()
    root.iconbitmap("./images/cu.ico")
    root.title("Call Option on Min of two Risky Assets--Yu Bai")
    main = MainView(root)
    main.pack(side="top", fill="both", expand=True)
    root.wm_geometry('900x500+100+100')
    root.mainloop()
