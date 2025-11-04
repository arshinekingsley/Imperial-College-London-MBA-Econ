import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# Historical TPM Data
data = {
    "Year": [2022, 2023, 2023, 2023, 2024, 2024, 2024, 2025, 2025, 2025, 2025],
    "Tier": ["Free", "Free", "Plus", "Enterprise", "Free", "Plus", "Enterprise",
             "Free", "Plus", "Pro", "Enterprise"],
    "Weighted_TPM": [80000, 200000, 350000, 900000, 250000, 500000, 950000,
                     287500, 550000, 775000, 925000]
}
df_tpm = pd.DataFrame(data)

# Historical TPM Plot
plt.figure(figsize=(10,6))
for tier in df_tpm['Tier'].unique():
    df_tier = df_tpm[df_tpm['Tier'] == tier]
    plt.plot(df_tier['Year'], df_tier['Weighted_TPM'], marker='o', label=tier)
plt.title('Weighted TPM per Tier (Historical 2022-2025)')
plt.xlabel('Year')
plt.ylabel('Weighted TPM (tokens/min)')
plt.legend()
plt.grid(True)
plt.show()

# Monte Carlo TPM (2025)
simulations_tpm = 1000
conversion_free_to_plus = 0.05
conversion_plus_to_pro = 0.01
variability = 0.10

tpm_2025 = {
    tier: df_tpm[(df_tpm['Year'] == 2025) & (df_tpm['Tier'] == tier)]['Weighted_TPM'].values[0]
    for tier in ["Free", "Plus", "Pro", "Enterprise"]
}

users_2025_tpm = {"Free": 1.0, "Plus": 0.5, "Pro": 0.3, "Enterprise": 0.1}
tiers = ["Free", "Plus", "Pro", "Enterprise"]
mc_tpm = {tier: [] for tier in tiers}

for tier in tiers:
    tier_sim = []
    for sim in range(simulations_tpm):
        tpm = tpm_2025[tier]
        users = users_2025_tpm[tier]

        if tier == "Free":
            users_after = users * (1 - conversion_free_to_plus)
            tpm_simulated = tpm * (users_after / users)
        elif tier == "Plus":
            users_after = users + users_2025_tpm["Free"] * conversion_free_to_plus
            users_after -= users * conversion_plus_to_pro
            tpm_simulated = tpm * (users_after / users)
        elif tier == "Pro":
            users_after = users + users_2025_tpm["Plus"] * conversion_plus_to_pro
            tpm_simulated = tpm * (users_after / users)
        else:
            tpm_simulated = tpm

        tpm_simulated *= np.random.normal(1, variability)
        tier_sim.append(tpm_simulated)
    mc_tpm[tier] = np.array(tier_sim)

# Plot Monte Carlo TPM
plt.figure(figsize=(10,6))
for tier, samples in mc_tpm.items():
    plt.hist(samples, bins=50, alpha=0.5, label=tier)
plt.title('Monte Carlo TPM Distributions (2025)')
plt.xlabel('Weighted TPM (tokens/min)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Revenue Simulation with Economic Structure
YEARS = np.arange(2025, 2035)
N_SIM = 5000

# Baseline revenue ($B)
revenue_2025 = {"Plus": 4.8, "Pro": 1.0, "Enterprise": 3.2}
price = {"Plus": 20, "Pro": 200, "Enterprise": 30000}

users_2025_revenue = {
    "Plus": revenue_2025["Plus"] * 1e9 / (price["Plus"] * 12),
    "Pro": revenue_2025["Pro"] * 1e9 / (price["Pro"] * 12),
    "Enterprise": revenue_2025["Enterprise"] * 1e9 / (price["Enterprise"] * 12)
}
enterprise_orgs_2025 = users_2025_revenue["Enterprise"]

# Churn & conversion
churn = {"Free": 0.10, "Plus": 0.08, "Pro": 0.12}
conv_FP_mean, conv_PP_mean = 0.05, 0.01

# Lerner index assumptions
elasticity_paid = {"Plus": -0.25, "Pro": -0.2, "Enterprise": -0.15}
cost_decline = 0.10
price_growth_base = {"Plus": 0.04, "Pro": 0.03, "Enterprise": 0.05}

network_effect_factor = 0.00001  # Free -> Paid network effect

# Scenarios
scenarios = {
    "baseline": {"growth": 0.15, "price_growth": 1.0},
    "bull": {"growth": 0.25, "price_growth": 1.2},
    "bear": {"growth": 0.08, "price_growth": 0.8}
}

def simulate_revenue_scenario(scenario):
    growth = scenarios[scenario]["growth"]
    price_growth_multiplier = scenarios[scenario]["price_growth"]
    results = []

    for sim in range(N_SIM):
        # Initialize users
        users_free = 800e6 * 0.57
        users_plus = users_2025_revenue["Plus"]
        users_pro = users_2025_revenue["Pro"]
        users_ent = enterprise_orgs_2025

        price_plus, price_pro, price_ent = price.values()
        cost_plus = price_plus * (1 - (-1 / elasticity_paid["Plus"]))
        cost_pro = price_pro * (1 - (-1 / elasticity_paid["Pro"]))
        cost_ent = price_ent * (1 - (-1 / elasticity_paid["Enterprise"]))

        for year in YEARS:
            # Stochastic churn/conversion
            f_FP = np.clip(np.random.normal(conv_FP_mean, 0.01), 0, 0.1)
            f_PP = np.clip(np.random.normal(conv_PP_mean, 0.002), 0, 0.05)
            churn_F = np.clip(np.random.normal(churn["Free"], 0.02), 0, 0.2)
            churn_P = np.clip(np.random.normal(churn["Plus"], 0.01), 0, 0.15)
            churn_R = np.clip(np.random.normal(churn["Pro"], 0.02), 0, 0.25)

            # User growth
            users_free *= (1 + growth)
            users_plus *= (1 + growth * 0.5)
            users_pro *= (1 + growth * 0.2)
            users_ent *= (1 + 0.10)

            # Conversions
            new_plus_from_free = users_free * f_FP
            new_pro_from_plus = users_plus * f_PP
            users_free = users_free * (1 - f_FP - churn_F)
            users_plus = users_plus * (1 - f_PP - churn_P) + new_plus_from_free
            users_pro = users_pro * (1 - churn_R) + new_pro_from_plus

            # Network effect
            users_plus_effective = users_plus + users_free * network_effect_factor
            users_pro_effective = users_pro + users_free * network_effect_factor

            # Price updates
            price_plus *= (1 + price_growth_base["Plus"] * price_growth_multiplier)
            price_pro *= (1 + price_growth_base["Pro"] * price_growth_multiplier)
            price_ent *= (1 + price_growth_base["Enterprise"] * price_growth_multiplier)

            # Cost decline
            cost_plus *= (1 - cost_decline)
            cost_pro *= (1 - cost_decline)
            cost_ent *= (1 - cost_decline)

            # Revenues
            revenue_plus = users_plus_effective * price_plus * 12
            revenue_pro = users_pro_effective * price_pro * 12
            revenue_ent = users_ent * price_ent * 12
            total_revenue = revenue_plus + revenue_pro + revenue_ent

            # Costs & profit
            total_cost = (users_plus_effective * cost_plus +
                          users_pro_effective * cost_pro +
                          users_ent * cost_ent) * 12
            profit = total_revenue - total_cost

            results.append({
                "Year": year,
                "Scenario": scenario,
                "Sim": sim,
                "Revenue": total_revenue / 1e9,
                "Profit": profit / 1e9,
                "Users_Plus": users_plus_effective / 1e6,
                "Users_Pro": users_pro_effective / 1e6,
                "Enterprises": users_ent
            })
    return pd.DataFrame(results)

# Run Monte Carlo revenue
df_rev_all = pd.concat([simulate_revenue_scenario(s) for s in scenarios.keys()])

# Revenue Summary & Plot
summary_rev = (
    df_rev_all.groupby(["Scenario", "Year"])["Revenue"]
    .agg(["mean", lambda x: np.percentile(x, 5), lambda x: np.percentile(x, 95)])
    .reset_index()
)
summary_rev.columns = ["Scenario", "Year", "Mean", "P5", "P95"]

plt.figure(figsize=(10,6))
for s in scenarios.keys():
    subset = summary_rev[summary_rev["Scenario"] == s]
    plt.plot(subset["Year"], subset["Mean"], label=f"{s} mean")
    plt.fill_between(subset["Year"], subset["P5"], subset["P95"], alpha=0.15)
plt.title("ChatGPT Revenue Forecast (Economic Model with Network Effects & Lerner)")
plt.xlabel("Year")
plt.ylabel("Revenue ($B)")
plt.legend()
plt.grid(True)
plt.show()

# Probability of $100B by 2027
target_year = 2027
target_revenue = 100
prob_hit_100 = (df_rev_all[df_rev_all["Year"] == target_year]["Revenue"] > target_revenue).mean() * 100
print(f"\nProbability of hitting $100B revenue by 2027: {prob_hit_100:.2f}%")
