# Generate tables and plots for paper
input_file = 'c:/data/AgentData/Simple/evaluate_all.txt'
output_latex_table = 'c:/data/AgentData/Simple/evaluate_all_latex.txt'

#input_file = 'c:/data/AgentData/Nature/evaluate_all.txt'
#output_latex_table = 'c:/data/AgentData/Nature/evaluate_all_latex.txt'


#df = read.csv('c:/data/AgentData/Simple/evaluate_all.txt')
df = read.csv(input_file)
unique_x = unique(df$x)
unique_power = unique(df$power)
unique_threshold = unique(df$threshold)

df_selected = df[,c("x","power","threshold","acc")]
tab_res = matrix(0, nrow(df_selected) / length(unique_x), length(unique_x))

#####################################
#Get unique algorithms
#####################################
df_help = df_selected[df_selected$x == 48,]
row_names = c()
for (a in 1:nrow(df_help))
{
  row_name = ""
  if (df_help[a,]$power == -1) 
  {
    row_name = "Naive"
  } 
  else {
    row_name = paste("$\\alpha$=", df_help[a,]$power, ";t=", df_help[a,]$threshold,sep = "")
  }
  row_names = c(row_names, row_name)
}
column_id = 1
for (xx in unique_x)
{
  df_help = df_selected[df_selected$x == xx,]
  for (row_id in 1:nrow(df_help))
  {
    tab_res[row_id, column_id] = df_help$acc[row_id]
  }
  column_id = column_id + 1
}
colnames(tab_res) = unique_x
rownames(tab_res) = row_names
tab_res_percent = round(tab_res / 3000, digits = 3)

######################################################
#Write to file
######################################################
sink(output_latex_table)
a = 1
for (cn in c("", colnames(tab_res_percent)))
{
  if (a > 1) cat(" & ")
  cat(paste("\\textbf{", cn, "} "))
  a = a + 1
}
cat("\\\\\n")
a = 1
for (rn in rownames(tab_res_percent))
{
  cat(paste("\\textbf{", rn, "} "))
  for (b in 1:ncol(tab_res_percent))
  {
    cat(" & ")
    cat(paste(tab_res_percent[a, b]))
  }
  cat("\\\\\n")
  a = a + 1
}

sink()

####################################################################################

path_dir = "c:/publikacje/PPO Entropy/data/"
files_list = list("scene_kameranature_nature_48x48_hidden16_01_SimpleCollector.csv",
  "scene_kameranature_nature_56x56_hidden16_01_SimpleCollector.csv",
  "scene_kameranature_nature_64x64_hidden16_01_SimpleCollector.csv",
  "scene_kameranature_nature_96x96_hidden16_01_SimpleCollector.csv",
  "scene_kameranature_nature_128x128_hidden16_01_SimpleCollector.csv",
  "scene_kameranature_nature_192x192_hidden16_01_SimpleCollector.csv",
  "scene_kameranature_simple_48x48_hidden16_01_SimpleCollector.csv",
  "scene_kameranature_simple_56x56_hidden16_01_SimpleCollector.csv",
  "scene_kameranature_simple_64x64_hidden16_01_SimpleCollector.csv",
  "scene_kameranature_simple_96x96_hidden16_01_SimpleCollector.csv",
  "scene_kameranature_simple_128x128_hidden16_01_SimpleCollector.csv",
  "scene_kameranature_simple_192x192_hidden16_01_SimpleCollector.csv")

alg_settings = list("Nature, 48x48",
                  "Nature, 56x56",
                  "Nature, 64x64",
                  "Nature, 96x96",
                  "Nature, 128x128",
                  "Nature, 192x192",
                  "Simple, 48x48",
                  "Simple, 56x56",
                  "Simple, 64x64",
                  "Simple, 96x96",
                  "Simple, 128x128",
                  "Simple, 192x192")

my_df_list = list()
r_color <- colors()
max_step = 75
for (a in 1:length(files_list))
{
  my_path = paste(path_dir, files_list[[a]], sep = "")
  my_df_list[[a]] = read.csv(my_path)
}
plot(my_df_list[[1]]$Step[1:max_step], my_df_list[[1]]$Value[1:max_step], type = "l", col=r_color[10 * 1], 
  xlab = "Step", ylab = "Cumulative Reward", ylim = c(-1,1), main = "Cumulative Reward value during PPO")
my_colors = r_color[10 * 1]
for (a in 2:length(files_list))
{
  lines(my_df_list[[a]]$Step[1:max_step], my_df_list[[a]]$Value[1:max_step], type = "l", col=r_color[10 * a])
  my_colors = c(my_colors, r_color[10 * a])
}
legend(10^6+100000, 0.3, legend=alg_settings, title="Network",
       col=my_colors, lty=rep(1,12), cex=0.8)

