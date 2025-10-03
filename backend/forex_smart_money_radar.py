import numpy as np
import pandas as pd
from datetime import datetime
import time
import requests
from colorama import init, Fore, Back, Style
import warnings
warnings.filterwarnings('ignore')

# Инициализация на цветен изход
init(autoreset=True)

class ForexSmartMoneyRadar:
    """
    Математически анализатор на Forex пазара
    Открива арбитражни възможности и следи smart money flow
    """
    
    def __init__(self):
        # 8-те основни валути
        self.major_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']
        
        # 28-те основни валутни двойки
        self.major_pairs = []
        for i in range(len(self.major_currencies)):
            for j in range(i+1, len(self.major_currencies)):
                self.major_pairs.append(f"{self.major_currencies[i]}/{self.major_currencies[j]}")
        
        # Праг за арбитражна възможност (в проценти)
        self.arbitrage_threshold = 0.1  # 0.1% разлика
        
        # Праг за манипулация/stop hunt (в проценти) 
        self.manipulation_threshold = 0.3  # 0.3% разлика
        
        # Съхранение на последните цени
        self.prices = {}
        self.calculated_prices = {}
        self.discrepancies = {}
        
    def get_mock_prices(self):
        """
        Симулирани цени за демонстрация
        В реална среда тук ще се свържете с брокер API
        """
        # Базови курсове спрямо USD (примерни)
        base_rates = {
            'EUR/USD': 1.3131,
            'GBP/USD': 1.5652,
            'USD/JPY': 137.01,
            'USD/CHF': 0.9192,
            'USD/CAD': 1.0870,
            'AUD/USD': 1.4065,
            'NZD/USD': 1.2890
        }
        
        # Добавяме малък шум за симулация на движение
        for pair, rate in base_rates.items():
            noise = np.random.normal(0, 0.0001)  # Малък случаен шум
            self.prices[pair] = rate * (1 + noise)
        
        # Изчисляваме кросовете
        self._calculate_cross_rates()
        
        # Добавяме изкуствена аномалия за демонстрация
        if np.random.random() > 0.7:  # 30% шанс за аномалия
            random_pair = np.random.choice(list(self.prices.keys()))
            self.prices[random_pair] *= 1.003  # 0.3% отклонение
            
    def _calculate_cross_rates(self):
        """
        Изчислява всички кросови курсове базирани на USD двойките
        """
        # EUR кросове
        if 'EUR/USD' in self.prices and 'GBP/USD' in self.prices:
            self.prices['EUR/GBP'] = self.prices['EUR/USD'] / self.prices['GBP/USD']
        if 'EUR/USD' in self.prices and 'USD/JPY' in self.prices:
            self.prices['EUR/JPY'] = self.prices['EUR/USD'] * self.prices['USD/JPY']
        if 'EUR/USD' in self.prices and 'USD/CHF' in self.prices:
            self.prices['EUR/CHF'] = self.prices['EUR/USD'] * self.prices['USD/CHF']
        if 'EUR/USD' in self.prices and 'USD/CAD' in self.prices:
            self.prices['EUR/CAD'] = self.prices['EUR/USD'] * self.prices['USD/CAD']
        if 'EUR/USD' in self.prices and 'AUD/USD' in self.prices:
            self.prices['EUR/AUD'] = self.prices['EUR/USD'] / self.prices['AUD/USD']
        if 'EUR/USD' in self.prices and 'NZD/USD' in self.prices:
            self.prices['EUR/NZD'] = self.prices['EUR/USD'] / self.prices['NZD/USD']
            
        # GBP кросове
        if 'GBP/USD' in self.prices and 'USD/JPY' in self.prices:
            self.prices['GBP/JPY'] = self.prices['GBP/USD'] * self.prices['USD/JPY']
        if 'GBP/USD' in self.prices and 'USD/CHF' in self.prices:
            self.prices['GBP/CHF'] = self.prices['GBP/USD'] * self.prices['USD/CHF']
        if 'GBP/USD' in self.prices and 'USD/CAD' in self.prices:
            self.prices['GBP/CAD'] = self.prices['GBP/USD'] * self.prices['USD/CAD']
        if 'GBP/USD' in self.prices and 'AUD/USD' in self.prices:
            self.prices['GBP/AUD'] = self.prices['GBP/USD'] / self.prices['AUD/USD']
        if 'GBP/USD' in self.prices and 'NZD/USD' in self.prices:
            self.prices['GBP/NZD'] = self.prices['GBP/USD'] / self.prices['NZD/USD']
            
        # Други важни кросове
        if 'AUD/USD' in self.prices and 'USD/JPY' in self.prices:
            self.prices['AUD/JPY'] = self.prices['AUD/USD'] * self.prices['USD/JPY']
        if 'NZD/USD' in self.prices and 'USD/JPY' in self.prices:
            self.prices['NZD/JPY'] = self.prices['NZD/USD'] * self.prices['USD/JPY']
        if 'USD/CAD' in self.prices and 'USD/JPY' in self.prices:
            self.prices['CAD/JPY'] = self.prices['USD/JPY'] / self.prices['USD/CAD']
        if 'USD/CHF' in self.prices and 'USD/JPY' in self.prices:
            self.prices['CHF/JPY'] = self.prices['USD/JPY'] / self.prices['USD/CHF']
            
    def check_triangular_arbitrage(self):
        """
        Проверява за триангуларен арбитраж между 3 валути
        """
        arbitrage_opportunities = []
        
        # Проверяваме EUR-USD-GBP триъгълник
        if all(pair in self.prices for pair in ['EUR/USD', 'GBP/USD', 'EUR/GBP']):
            # Директен път: EUR -> USD -> GBP -> EUR
            calculated_eur_gbp = self.prices['EUR/USD'] / self.prices['GBP/USD']
            actual_eur_gbp = self.prices['EUR/GBP']
            
            discrepancy = abs((calculated_eur_gbp - actual_eur_gbp) / actual_eur_gbp * 100)
            
            if discrepancy > self.arbitrage_threshold:
                arbitrage_opportunities.append({
                    'triangle': 'EUR-USD-GBP',
                    'calculated': calculated_eur_gbp,
                    'actual': actual_eur_gbp,
                    'discrepancy': discrepancy,
                    'type': 'ARBITRAGE' if discrepancy < self.manipulation_threshold else 'MANIPULATION'
                })
        
        # Проверяваме EUR-USD-JPY триъгълник
        if all(pair in self.prices for pair in ['EUR/USD', 'USD/JPY', 'EUR/JPY']):
            calculated_eur_jpy = self.prices['EUR/USD'] * self.prices['USD/JPY']
            actual_eur_jpy = self.prices['EUR/JPY']
            
            discrepancy = abs((calculated_eur_jpy - actual_eur_jpy) / actual_eur_jpy * 100)
            
            if discrepancy > self.arbitrage_threshold:
                arbitrage_opportunities.append({
                    'triangle': 'EUR-USD-JPY',
                    'calculated': calculated_eur_jpy,
                    'actual': actual_eur_jpy,
                    'discrepancy': discrepancy,
                    'type': 'ARBITRAGE' if discrepancy < self.manipulation_threshold else 'MANIPULATION'
                })
                
        return arbitrage_opportunities
    
    def calculate_currency_strength(self):
        """
        Изчислява относителната сила на всяка валута
        """
        strength_matrix = pd.DataFrame(0.0, 
                                      index=self.major_currencies, 
                                      columns=self.major_currencies)
        
        # Попълваме матрицата с курсове
        for pair, rate in self.prices.items():
            if '/' in pair:
                base, quote = pair.split('/')
                if base in self.major_currencies and quote in self.major_currencies:
                    strength_matrix.loc[base, quote] = rate
                    strength_matrix.loc[quote, base] = 1 / rate
        
        # Изчисляваме средната сила на всяка валута
        currency_strength = {}
        for currency in self.major_currencies:
            # Средна стойност спрямо всички други валути
            values = []
            for other in self.major_currencies:
                if currency != other and strength_matrix.loc[currency, other] != 0:
                    values.append(strength_matrix.loc[currency, other])
            
            if values:
                # Нормализираме към USD = 100
                if currency == 'USD':
                    currency_strength[currency] = 100
                else:
                    avg_rate = np.mean(values)
                    currency_strength[currency] = 100 / avg_rate if currency in ['JPY', 'CHF', 'CAD'] else 100 * avg_rate
        
        return currency_strength
    
    def detect_smart_money_flow(self):
        """
        Открива къде се движат умните пари
        """
        strength = self.calculate_currency_strength()
        
        # Сортираме валутите по сила
        sorted_strength = sorted(strength.items(), key=lambda x: x[1], reverse=True)
        
        # Най-силни и най-слаби валути
        strongest = sorted_strength[:2]
        weakest = sorted_strength[-2:]
        
        return {
            'strongest': strongest,
            'weakest': weakest,
            'recommendation': f"КУПИ {strongest[0][0]}/{weakest[0][0]}"
        }
    
    def display_dashboard(self):
        """
        Показва красив дашборд в конзолата
        """
        print("\n" + "="*80)
        print(f"{Back.BLUE}{Fore.WHITE} FOREX SMART MONEY RADAR - {datetime.now().strftime('%H:%M:%S')} {Style.RESET_ALL}")
        print("="*80)
        
        # Проверка за арбитраж
        arbitrage = self.check_triangular_arbitrage()
        
        if arbitrage:
            print(f"\n{Fore.YELLOW}⚠️  ОТКРИТИ АНОМАЛИИ:{Style.RESET_ALL}")
            for opp in arbitrage:
                if opp['type'] == 'ARBITRAGE':
                    print(f"{Fore.GREEN}  💰 АРБИТРАЖ: {opp['triangle']}")
                    print(f"     Изчислен: {opp['calculated']:.4f}")
                    print(f"     Реален:   {opp['actual']:.4f}")
                    print(f"     Разлика:  {opp['discrepancy']:.2f}%{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}  🚨 ВЪЗМОЖНА МАНИПУЛАЦИЯ: {opp['triangle']}")
                    print(f"     Разлика: {opp['discrepancy']:.2f}%")
                    print(f"     ВНИМАНИЕ: Stop hunt вероятен!{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.GREEN}✓ Няма открити аномалии{Style.RESET_ALL}")
        
        # Smart Money Flow
        smart_flow = self.detect_smart_money_flow()
        
        print(f"\n{Fore.CYAN}📊 ВАЛУТНА СИЛА:{Style.RESET_ALL}")
        print(f"{Fore.GREEN}  Най-силни валути:")
        for curr, strength in smart_flow['strongest']:
            print(f"    • {curr}: {strength:.1f}")
        
        print(f"{Fore.RED}  Най-слаби валути:")
        for curr, strength in smart_flow['weakest']:
            print(f"    • {curr}: {strength:.1f}")
        
        print(f"\n{Back.GREEN}{Fore.BLACK} ПРЕПОРЪКА: {smart_flow['recommendation']} {Style.RESET_ALL}")
        
        # Показваме няколко ключови курса
        print(f"\n{Fore.MAGENTA}📈 КЛЮЧОВИ КУРСОВЕ:{Style.RESET_ALL}")
        key_pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'EUR/GBP', 'EUR/JPY']
        for pair in key_pairs:
            if pair in self.prices:
                print(f"  {pair}: {self.prices[pair]:.4f}")
        
        print("\n" + "="*80)
    
    def run_live_monitoring(self, refresh_seconds=5, iterations=10):
        """
        Стартира мониторинг в реално време
        """
        print(f"\n{Back.GREEN}{Fore.BLACK} СТАРТИРАНЕ НА МОНИТОРИНГ... {Style.RESET_ALL}")
        print(f"Обновяване на всеки {refresh_seconds} секунди")
        print(f"Брой проверки: {iterations}")
        
        for i in range(iterations):
            # Получаваме нови цени
            self.get_mock_prices()
            
            # Показваме дашборда
            self.display_dashboard()
            
            if i < iterations - 1:
                print(f"\n{Fore.YELLOW}Следваща проверка след {refresh_seconds} секунди...{Style.RESET_ALL}")
                time.sleep(refresh_seconds)
        
        print(f"\n{Back.RED}{Fore.WHITE} МОНИТОРИНГЪТ ПРИКЛЮЧИ {Style.RESET_ALL}")


# ГЛАВНА ПРОГРАМА
if __name__ == "__main__":
    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════╗
║          FOREX SMART MONEY RADAR v1.0                         ║
║          Математически анализатор на валутни двойки           ║
║          Базиран на видеото: Forex and Math Secrets           ║
╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
    """)
    
    # Създаваме радара
    radar = ForexSmartMoneyRadar()
    
    # Избор на режим
    print("\nИзберете режим:")
    print("1. Еднократна проверка")
    print("2. Live мониторинг (10 проверки)")
    print("3. Непрекъснат мониторинг")
    
    choice = input("\nВашият избор (1-3): ")
    
    if choice == '1':
        radar.get_mock_prices()
        radar.display_dashboard()
    elif choice == '2':
        radar.run_live_monitoring(refresh_seconds=3, iterations=10)
    elif choice == '3':
        try:
            print("\n(Натиснете Ctrl+C за спиране)")
            while True:
                radar.get_mock_prices()
                radar.display_dashboard()
                time.sleep(3)
        except KeyboardInterrupt:
            print(f"\n{Back.RED}{Fore.WHITE} МОНИТОРИНГЪТ Е СПРЯН {Style.RESET_ALL}")
    else:
        print("Невалиден избор!")
        
    print(f"\n{Fore.GREEN}Благодаря, че използвахте Smart Money Radar!{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Успех в търговията!{Style.RESET_ALL}")
