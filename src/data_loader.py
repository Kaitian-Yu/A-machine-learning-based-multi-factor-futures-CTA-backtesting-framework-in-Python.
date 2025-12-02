def load_data(folder):
  folder = folder
  dfs = []
  
  for file in os.listdir(folder):
      if file.endswith('.csv'):
          path = os.path.join(folder, file)
          sym = file.replace('.csv', '')
          df = pd.read_csv(path)
          df['symbol'] = sym
          dfs.append(df)
  
  data = pd.concat(dfs).sort_values(['symbol', 'time'])
  
  # If time is a string, convert it to a time (optional)
  try:
      data['time'] = pd.to_datetime(data['time'])
  except Exception:
      pass
  
  print("Raw data:")
  print(data.head())
  print(data.shape)
