# Plot birthweight vs body length for CPP data (collaborative perinatal project data)

module CPPdata
using PyPlot
draw() = (pause(0.001); get_current_fig_manager()[:window][:raise_]())

target_name = "V_BWGT"
predictor_name = "V_BLENG"

# Load data
data = readdlm("datasets/CPP/finaldde.csv",',')
names = vec(data[1,:])
values = data[2:end,:]
target_index = findfirst(names.==target_name)
predictor_index = findfirst(names.==predictor_name)
missing = vec(maximum(values[:,[predictor_index,target_index]].==".", 2))
y = convert(Array{Float64,1}, values[!missing,target_index])
x = convert(Array{Float64,1}, values[!missing,predictor_index])
n = length(y)
println("Removed $(sum(missing)) records with missing entries.")


# Plot
figure(1,figsize=(8,3.2)); clf(); hold(true)
subplots_adjust(bottom=0.2)
plot(x,y/1000,"b.",markersize=3)
xlabel("\$\\mathrm{body\\,length\\,(cm)}\$",fontsize=17)
ylabel("\$\\mathrm{birthweight\\,(kg)}\$",fontsize=16)
xlim(35,60)
draw()
savefig("cpp-length-vs-weight.png",dpi=150)

end # module


