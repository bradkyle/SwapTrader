
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from datetime import datetime
from pandas.plotting import register_matplotlib_converters
import pandas as pd

if __name__ == '__main__':
    from util import *
    from constants import *
else:
    from env.util import *
    from env.constants import *


style.use('ggplot')
register_matplotlib_converters()

VOLUME_CHART_HEIGHT = 0.33
POSITION_CHART_HEIGHT = 0.33


class SwapTradingGraph:
    """A Bitcoin trading visualization using matplotlib made to render OpenAI gym environments"""

    def __init__(self, df, account_history):
        self.df = df
        self.df.reset_index(inplace=True)
        self.df.sort_values(by='window', inplace=True)
        self.df['date'] = pd.to_datetime(df['window'], unit='ms')
        self.df['date'].dt.round('15min') 
        self.df = self.df.sort_values('date')

        # print(np.transpose(account_history))

        # Create a figure on screen and set the title
        self.fig = plt.figure()

        # Create top subplot for net worth axis
        self.net_worth_ax = plt.subplot2grid(
            (6, 1), (0, 0), rowspan=2, colspan=1)

        # Create bottom subplot for shared price/volume axis
        self.price_ax = plt.subplot2grid(
            (6, 1), (2, 0), rowspan=8, colspan=1, sharex=self.net_worth_ax)

        # Create a new axis for volume which shares its x-axis with price
        self.volume_ax = self.price_ax.twinx()

        # Create bottom subplot for shared price/volume axis
        # self.position_ax = plt.subplot2grid((6, 1), (2, 0), rowspan=8, colspan=1, sharex=self.net_worth_ax)

        # Add padding to make graph easier to view
        plt.subplots_adjust(left=0.11, bottom=0.24,
                            right=0.90, top=0.90, wspace=0.2, hspace=0.2)

        # Show the graph without blocking the rest of the program
        plt.show(block=False)

    def _render_price(self, step_range, dates, current_step):
        self.price_ax.clear()

        # Plot price using candlestick graph from mpl_finance
        self.price_ax.plot(dates, self.df['price'].values[step_range], color="black")

        last_date = self.df['date'].values[current_step]
        last_close = self.df['price'].values[current_step]
        last_high = self.df['price'].values[current_step]

        # Print the current price to the price axis
        self.price_ax.annotate('{0:.2f}'.format(last_close), (last_date, last_close),
                               xytext=(last_date, last_high),
                               bbox=dict(boxstyle='round',
                               fc='w', ec='k', lw=1),
                               color="black",
                               fontsize="small")

        # Shift price axis up to give volume chart space
        ylim = self.price_ax.get_ylim()
        self.price_ax.set_ylim(ylim[0] - (ylim[1] - ylim[0]) * VOLUME_CHART_HEIGHT, ylim[1])

    def _render_position(self, step_range, dates, lng, sht):
        self.position_ax.clear()

        lngs = np.array(lng['values'][step_range])
        shts = np.array(sht['values'][step_range])

        self.position_ax.plot(dates, lngs,  color='green')
        self.position_ax.fill_between(dates, lngs, color='green', alpha=0.5)

        self.position_ax.plot(dates, shts,  color='red')
        self.position_ax.fill_between(dates, shts, color='red', alpha=0.5)

        self.position_ax.set_ylim(0, max([max(lngs), max(shts)]) / POSITION_CHART_HEIGHT)
        self.position_ax.yaxis.set_ticks([])

    def _render_trades(self, step_range, trades):
        for trade in trades:
            if trade['step'] in range(sys.maxsize)[step_range]:
                date = self.df['date'].values[trade['step']]
                price = self.df['price'].values[trade['step']]

                if trade['type'] == 'buy':
                    color = 'g'
                else:
                    color = 'r'

                self.price_ax.annotate(' ', (date, price),
                                       xytext=(date, price),
                                       size="large",
                                       arrowprops=dict(arrowstyle='simple', facecolor=color))

    def _render_account(self, step_range, dates):
        self.volume_ax.clear()

        long_position = np.array(self.account_df['long_position'].values[step_range])
        short_position = np.array(self.account_df['short_position'].values[step_range])

        self.volume_ax.plot(dates, long_position,  color='blue')
        self.volume_ax.fill_between(dates, long_position, color='blue', alpha=0.5)

        self.volume_ax.plot(dates, short_position,  color='red')
        self.volume_ax.fill_between(dates, short_position, color='red', alpha=0.5)

        self.volume_ax.set_ylim(0, max([max(long_position), max(short_position)]) / VOLUME_CHART_HEIGHT)
        self.volume_ax.yaxis.set_ticks([])


    def render(self, current_step, net_worths, account_history, window_size=200):        
        print(np.array(account_history).shape)
        self.account_df = pd.DataFrame(np.array(account_history), columns=list(Account._fields))
        
        net_worth = np.round(net_worths[-1],5)
        initial_net_worth = np.round(net_worths[0], 5)
        profit_percent = np.round((net_worth - initial_net_worth) / initial_net_worth * 100, 2)

        self.fig.suptitle('Net worth: $' + str(net_worth) + ' | Profit: ' + str(profit_percent) + '%')

        window_start = max(current_step - window_size, 0)
        step_range = slice(window_start, current_step + 1)
        dates = self.df['date'].values[step_range]

        self._render_price(step_range, dates, current_step)
        self._render_account(step_range, dates)
        # self._render_trades(step_range, trades)

        date_labels = self.df['date'].values[step_range]

        self.price_ax.set_xticklabels(date_labels, rotation=45, horizontalalignment='right')

        # Hide duplicate net worth date labels
        plt.setp(self.net_worth_ax.get_xticklabels(), visible=False)

        # Necessary to view frames before they are unrendered
        plt.pause(0.001)

    def price(self):
        plt.price()
