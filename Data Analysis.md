
---

#  Python & Data Analysis Notes

---

##  Matplotlib

```python
import matplotlib.pyplot as plt
```

1. **Line chart:**

   ```python
   plt.plot(x_values, y_values)
   plt.show()
   ```

2. **Scatter plot:**

   ```python
   plt.scatter(x_values, y_values)
   ```

3. **Logarithmic scale:**

   ```python
   plt.xscale('log')
   plt.yscale('log')
   ```

4. **Histogram:**

   ```python
   plt.hist(values, bins=10)
   ```

5. **Clear figure:**

   ```python
   plt.clf()
   ```

### Graph Customization

```python
plt.xlabel("X-axis label")
plt.ylabel("Y-axis label")
plt.title("Graph Title")
plt.yticks([0, 1, 2], ["Low", "Medium", "High"])
plt.grid(True)
```

* Line plot → show trends over time
* Scatter plot → show relationships

---

##  Dictionaries

1. **Definition:** Access elements without using an index.
2. **Syntax:**

   ```python
   my_dict = {key1: value1, key2: value2}
   ```
3. **Access elements:**

   ```python
   my_dict[key]
   ```
4. **Add element:**

   ```python
   my_dict['new_key'] = new_value
   ```
5. **Delete element:**

   ```python
   del my_dict['key']
   ```

### Pandas DataFrame from Dictionary

```python
import pandas as pd
df = pd.DataFrame(my_dict)
df.index = ["BR","RU","IN"]
```

### Import CSV

```python
df = pd.read_csv("file.csv")
```

### Access Columns & Rows

```python
df["column_name"]           # Single column
df[["col1","col2"]]         # Multiple columns
df[start:end]               # Slice rows
df.loc["label_name"]        # Label-based
df.iloc[0:5,1:4]            # Position-based
```

---

## Operators

* Use `np.logical_and()`, `np.logical_or()`, `np.logical_not()` instead of `and/or` in NumPy arrays:

```python
arr[np.logical_and(arr>1, arr<5)]
```

---

## Loops

### Basic Loops

```python
while condition:
    expression

for var in sequence:
    expression
```

### Enumerate

```python
fam = [1.73, 1.68, 1.71]
for index, height in enumerate(fam):
    print(f"index {index}: {height}")
```

### Dictionary Iteration

```python
for key, value in world.items():
    print(key, value)
```

### NumPy Iteration

```python
for val in np.nditer(array):
    print(val)
```

### Pandas Iteration

```python
for label, row in df.iterrows():
    print(label, row["column_name"])
```

* Add column via `loc` or `apply`:

```python
df.loc[label, "new_col"] = len(row["column_name"])
df["new_col"] = df["column"].apply(len)
```

---

## Random Numbers

```python
np.random.seed(10)
np.random.rand()
np.random.rand(3)           # 3 random numbers
np.random.rand(2,3)         # 2x3 array
```

---

## DataFrame Operations

### Exploring

```python
df.head()
df.sort_values("col", ascending=False)
df[["col1","col2"]]
df["new_col"] = df["existing_col"]
```

### Summarizing

```python
df["height_cm"].mean()
df["weight_kg"].cumsum()
df["weight_kg"].cummax()
```

### Counting

```python
df["breed"].value_counts()
df.drop_duplicates(subset="name")
```

### Grouping

```python
df.groupby("color")["weight_kg"].mean()
df.groupby(["color","breed"])["weight_kg"].agg([min,max,sum])
```

### Pivot Table

```python
df.pivot_table(values="weight_kg", index="color", aggfunc="mean")
df.pivot_table(values="weight_kg", index="color", columns="breed", fill_value=0, margins=True)
```

### Indexing

```python
df_ind = df.set_index("name")
df_ind.reset_index(drop=True)
df_ind.loc["labrador"]
df_ind3.loc[["labrador","chihuahua"]]
```

### Slicing with `.loc` & `.iloc`

```python
df_srt.loc[("labrador","brown"):("schnauzer","grey")]
df_srt.loc[:,"name":"height_cm"]
df.iloc[2:5,1:4]
```

---

## Visualization

### Matplotlib

```python
df["col"].hist(bins=10)
df.plot(kind="bar", x="label", y="value")
df.plot(x="x_label", y="y_label", kind="scatter")
plt.legend(["Label1","Label2"])
```

### Seaborn

```python
import seaborn as sns

sns.scatterplot(x="x", y="y", data=df, hue="category")
sns.catplot(x="x", y="y", data=df, kind="box", sym="")
sns.relplot(x="x", y="y", data=df, kind="scatter", row="smoker", col="time")
sns.kdeplot(data=df, x="column", hue="category")
sns.heatmap(df.corr(), annot=True)
```

### Styling

```python
sns.set_style("whitegrid")
sns.set_palette("RdBu")
sns.set_context("talk")
plt.xticks(rotation=90)
```

---

## Missing Values

```python
df.isna()
df.isna().sum()
df.dropna()
df.fillna(0)
```

---

## Reading & Writing CSV

```python
df.to_csv("new_file.csv")
```

---

## Joins & Merges

```python
df1.merge(df2, on="key")           # Inner join
df1.merge(df2, on="key", how="left")
df1.merge(df2, left_on="id", right_on="id", how="right")
pd.concat([df1, df2])
pd.merge_ordered(df1, df2, on="date")
pd.merge_asof(df1, df2, on="date", direction="forward")
```

---

## Filtering & Query

```python
df.query('column > 90')
df.query('column1 > 90 and column2 < 140')
df["column"].str.contains("Scientist|AI")
```

---

## Statistics

```python
import statistics
statistics.mode(df["column"])
np.var(df["column"], ddof=1)
np.std(df["column"], ddof=1)
np.quantile(df["column"], 0.5)
from scipy.stats import iqr
iqr(df["column"])
```

### Probability Distributions

```python
from scipy.stats import uniform, binom, poisson, expon, norm

uniform.rvs(0,5,size=10)
binom.rvs(n=10, p=0.5, size=10)
poisson.pmf(5,8)
expon.cdf(1, scale=2)
norm.cdf(x, mean, std)
```

---

## Time Series

```python
df['date'] = pd.to_datetime(df['date'])
df['2015'].info()
```

---


Do you want me to do that?
