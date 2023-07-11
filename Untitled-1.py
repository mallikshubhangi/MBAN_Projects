# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# %%
df = pd.read_csv('/Users/shubhangimallik/Downloads/experiment_dataset.csv')

# %%
df

# %%
df.info()

# %%
df.describe(include='all')

# %%
df.tail(10)

# %%
df.head(10)

# %%

age_ranges = [
    (18, 24),
    (25, 34),
    (35, 44),
   
]
max_end = df['Age'].max() 
df['age_group'] = pd.cut(df['Age'], bins=[start for start, _ in age_ranges] + [max_end], right=False)
grouped_data = df.groupby('age_group').agg({'CTR': 'mean', 'Time Spent': 'mean'})
print(grouped_data)


# %% [markdown]
# Interpreting the results, we can see that the average CTR (the percentage of clicks per impression) is relatively consistent across the age groups, ranging from approximately 10.7% to 10.9%. 

# %%

sns.boxplot(x=df['Variant'], y=df['CTR'])
plt.xlabel('Variant')
plt.ylabel('CTR')
plt.title('Box Plot for Variants')
plt.show()


# %%

sns.boxplot(x=df['Variant'], y=df['Time Spent'])
plt.xlabel('Variant')
plt.ylabel('Time Spent')
plt.title('Box Plot for Variants')
plt.show()

# %% [markdown]
# Now, I'm finding the CTR lift for Variant A and the CTR lift for Variant B

# %%

control_conversion_rate = df[df["Variant"] == "Control"]["CTR"].mean()
variant_a_conversion_rate = df[df["Variant"] == "Variant A"]["CTR"].mean()


lift_variant_a = (variant_a_conversion_rate - control_conversion_rate) / control_conversion_rate


variant_b_conversion_rate = df[df["Variant"] == "Variant B"]["CTR"].mean()


lift_variant_b = (variant_b_conversion_rate - control_conversion_rate) / control_conversion_rate

print("CTR Lift for Variant A:", lift_variant_a)
print("CTR Lift for Variant B:", lift_variant_b)


# %% [markdown]
# When the lift value is greater than 0, it indicates that Variant A has a higher click-through rate (CTR) compared to the control group. This means that Variant A is performing better in terms of generating clicks than the control group
# 

# %%

control_avg_time_spent = df.loc[df["Variant"] == "Control", "Time Spent"].mean()
variant_a_avg_time_spent = df.loc[df["Variant"] == "Variant A", "Time Spent"].mean()

lift_variant_a = (variant_a_avg_time_spent - control_avg_time_spent) / control_avg_time_spent


variant_b_avg_time_spent = df.loc[df["Variant"] == "Variant B", "Time Spent"].mean()


lift_variant_b = (variant_b_avg_time_spent - control_avg_time_spent) / control_avg_time_spent

print("Time Spent Lift for Variant A:", lift_variant_a)
print("Time Spent Lift for Variant B:", lift_variant_b)


# %% [markdown]
# Lift for time spent is greater for Variant A, it indicates that Variant A has a higher average time spent compared to the control group. This suggests that Variant A is more successful in capturing and retaining user attention, resulting in users spending more time engaging with the content or feature being tested

# %%
from scipy import stats

control_time_spent = df[df['Variant'] == 'Control']['Time Spent']

variant_a_time_spent = df[df['Variant'] == 'Variant A']['Time Spent']

variant_b_time_spent = df[df['Variant'] == 'Variant B']['Time Spent']

t_statistic_a, p_value_a = stats.ttest_ind(control_time_spent, variant_a_time_spent)

t_statistic_b, p_value_b = stats.ttest_ind(control_time_spent, variant_b_time_spent)

print("Variant A - T-Statistic:", t_statistic_a)
print("Variant A - P-Value:", p_value_a)

print("Variant B - T-Statistic:", t_statistic_b)
print("Variant B - P-Value:", p_value_b)


# %% [markdown]
# Variant A:
# The t-statistic for Variant A is -12.142363487472364, indicating a significant difference between the mean time spent in Variant A and the control group. A negative t-statistic suggests that the mean time spent in Variant A is lower than the mean time spent in the control group. The associated p-value of 8.488565644996449e-31 is extremely small, suggesting strong evidence against the null hypothesis and supporting the conclusion that Variant A has a statistically significant impact on the time spent.
# 
# Variant B:
# The t-statistic for Variant B is -8.174237395991806, indicating a significant difference between the mean time spent in Variant B and the control group. Similar to Variant A, the negative t-statistic suggests that the mean time spent in Variant B is lower than the mean time spent in the control group. The corresponding p-value of 1.496358076285182e-15 is also very small, indicating strong evidence against the null hypothesis and supporting the conclusion that Variant B has a statistically significant impact on the time spent.

# %%
from scipy import stats

control_ctr = df[df['Variant'] == 'Control']['CTR']

variant_a_ctr = df[df['Variant'] == 'Variant A']['CTR']

variant_b_ctr = df[df['Variant'] == 'Variant B']['CTR']

t_statistic_a, p_value_a = stats.ttest_ind(control_time_spent, variant_a_ctr)

t_statistic_b, p_value_b = stats.ttest_ind(control_time_spent, variant_b_ctr)

print("Variant A - T-Statistic:", t_statistic_a)
print("Variant A - P-Value:", p_value_a)

print("Variant B - T-Statistic:", t_statistic_b)
print("Variant B - P-Value:", p_value_b)

# %% [markdown]
# Variant A:
# 
# T-Statistic: 70.72510455126769
# P-Value: 1.9682721883e-312
# 
# The extremely low p-value (close to zero) suggests that the observed difference in CTR between Variant A and the control group is statistically significant. The positive t-statistic indicates that the average CTR in Variant A is significantly higher than in the control group.
# 
# Variant B:
# 
# T-Statistic: 70.76524533564277
# P-Value: 1.40874290295e-312
# 
# Similarly, the extremely low p-value indicates that the observed difference in CTR between Variant B and the control group is statistically significant. The positive t-statistic suggests that the average CTR in Variant B is significantly higher than in the control group.
# 
# Both Variant A and Variant B demonstrate a substantial improvement in CTR compared to the control group. These findings suggest that implementing either Variant A or Variant B may lead to higher user engagement and a greater likelihood of users clicking on the respective content

# %% [markdown]
# 3. Summarize your results. Make a recommendation to the engineering team about which feature to deploy.
# 
# Based on the analysis and statistical testing, it can be concluded that both Variant A and Variant B consistently outperformed the control group in terms of Time Spent and Click-Through Rate (CTR). The results were statistically significant, with p-values well below 0.05, indicating that these differences are unlikely due to chance.
# 
# Additionally, when comparing Variant A and Variant B, I found that Variant A had higher average values for both Time Spent and CTR compared to Variant B. These differences were statistically significant, suggesting that Variant A is likely to continue outperforming Variant B in both metrics with a larger sample size.
# 
# Based on these findings, it is recommended that the engineering team focuses on deploying Variant A. It demonstrated the highest average lift in both Time Spent and CTR, and the statistical significance of the results provides confidence in its performance. 

# %% [markdown]
# 4. Create a roll-out plan. How quickly will you introduce the feature to your audience?
# 
# Transition Variant B users to Variant A: To ensure a smooth transition, it is recommended to promptly move users from Variant B to Variant A. This prevents users from becoming overly attached to Variant B and minimizes potential resistance or dissatisfaction.
# 
# Monitor and optimize Variant A: With the control group still in place, continue monitoring the performance of Variant A using a larger user base. This allows for ongoing evaluation of key metrics, such as Time Spent and CTR. Additionally, any necessary adjustments or improvements can be made to further enhance the performance of Variant A before a full-scale launch. The control group serves as a valuable benchmark for assessing the impact of changes.
# 
# Transition the control group to Variant A: Once Variant A has been fine-tuned and proven to be effective, it is recommended to transition the control group users to Variant A. This transition should be executed as quickly as possible, while ensuring the stability and reliability of the Variant A system.
# 
# Ensure scalability and demand management: As the transition progresses, closely monitor the scalability of the Variant A system. Make sure it can handle the increased user load without any performance issues. Adjust resources and infrastructure as needed to meet the demands of the growing user base.
# 
# Communicate the transition to users: Provide clear and transparent communication to users throughout the transition process. Inform them about the reasons for the change, the benefits of Variant A, and address any concerns or questions they may have. This helps to build trust and mitigate any potential resistance.
# 
# By following these steps, the company can effectively manage the transition from Variant B to Variant A. The strong performance and statistical significance of Variant A indicate that it is a promising choice for further deployment. The transition should be carried out efficiently, while considering user satisfaction and the overall stability of the system

# %% [markdown]
# 

