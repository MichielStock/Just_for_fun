using PyPlot
xkcd()

struct Circle
    x::Float64
    y::Float64
    r::Float64
end

c1 = Circle(10, 3, 4)
c2 = Circle(2, 6, 4)
c3 = Circle(4, 6, 1)

function plot_circle(circle::Circle, ax, color="black")
    t = linspace(0, 2pi, 200)
    xvals = circle.r .* cos.(t) .+ circle.x
    yvals = circle.r .* sin.(t) .+ circle.y
    ax[:plot](xvals, yvals, color=color)
end

function generate_circles(n_circles::Int64, radius::Float64=1.0,
                            boxheight::Float64=20.0, boxwidth::Float64=30.0)
    circles = Array{Circle}(n_circles)
    for i in 1:n_circles
        x = rand() * (boxwidth - 2radius) + radius
        y = rand() * (boxheight - 2radius) + radius
        circles[i] = Circle(x, y, radius)
    end
    circles
end

function check_overlap_circles(circles::Array{Circle}, radius::Float64=1.0)
    n_circles = length(circles)
    overlap = Array{Bool}(n_circles)
    fill!(overlap, false)
    radiussq = radius^2
    ax[:set_xlim]([0, 30])
    ax[:set_ylim]([0, 20])
    for i in 2:n_circles
        for j in 1:(i-1)
            x1, y1 = circles[i].x, circles[i].y
            x2, y2 = circles[j].x, circles[j].y
            if (x1 - x2)^2 + (y1 - y2)^2 < 4radiussq
                overlap[i] = true
                overlap[j] = true
            end
        end
    end
    return overlap
end

function get_coordinates_circles(circles::Array{Circle})
    n_circles = length(circles)
    coordinates = Array{Float64}(n_circles, 2)
    for (i, circle) in enumerate(circles)
        coordinates[i,1] = circle.x
        coordinates[i,2] = circle.y
    end
    return coordinates
end

function simulate_circles(n_samples::Int64, n_circles::Int64)
    circles = Array{Circle}(n_circles)
    coordinates = Array{Float64}(n_samples * n_circles, 2)
    overlap = Array{Bool}(n_circles)
    n_succes = 0
    sim = 0
    while n_succes < n_samples
        sim = sim + 1
        circles[:] = generate_circles(n_circles)
        overlap[:] = check_overlap_circles(circles)
        if !any(overlap)
            n_succes = n_succes + 1
            coordinates[n_circles * (n_succes - 1) + 1 : n_circles * n_succes,:] = get_coordinates_circles(circles)
        end
    end
    return coordinates, n_succes / sim
end


# some testing
fig, ax = subplots()
ax[:set_aspect]("equal")
plot_circle(c1, ax)
plot_circle(c2, ax, "green")
savefig("circles_example")

n_circles = 10
n_example_simulations = 10

for sim in 1:n_example_simulations
    circles = generate_circles(n_circles)
    fig, ax = subplots()
    ax[:set_aspect]("equal")
    overlap = check_overlap_circles(circles)
    for (overlapping, circle) in zip(overlap, circles)
        if overlapping
            plot_circle(circle, ax, "red")
        else
            plot_circle(circle, ax, "black")
        end
    end
    savefig("simulation_$sim")
end


# Simulations

n_samples = 1000000

println("Doing similations...")

coordinates, success_rate = simulate_circles(n_samples, n_circles)
println("Finished! Success rate = $(success_rate * 100)%.")

fig, ax = subplots()
hist2D(coordinates[:,1], coordinates[:,2], normed=true, bins=50)
colorbar()
savefig("circle_histogram")
