using Random
#using GR
using Plots
using Distributions
using StatsBase
using NLopt
#using Calculus

Random.seed!(1);


d = MixtureModel(Normal[
   Normal(-2.0, 1.2),
   Normal(0.0, 1.0),
   Normal(3.0, 2.5)], [0.2, 0.5, 0.3]);
p_true = [0.2, -2.0, 1.2, 0.5, 0.0, 1.0, 0.3, 3.0, 2.5]

z = rand(d,10000);
gr()
histogram(z, nbins = 100, normalize = true)

function LogLikelihoodGaussianMixtureEstimate(p::Vector, grad::AbstractVector, x::Array{Float64})
   mod(length(p), 3) == 0 || error("length(p) must be divisible by 3")
   lh = 0.
   if length(grad) > 0
      gradient = zero(grad)
#      grad = zero(grad)
      for i in eachindex(grad)
         grad[i] = 0
      end
      for event in x
         ll = Array{Float64,1}(undef,Int64(length(p)/3))
         maxLLIndex = 1
         for i in 1:3:length(p)
            #amplitude = p[i]
            distance = p[i+1] - event
            sigma = p[i+2]
            norm = p[i]/sqrt(2π)/sigma
            myLLIndex = Int64( (i-1)/3 + 1 )
            ll[myLLIndex] = log(norm) - 0.5(distance/sigma)^2
            if ll[myLLIndex] > ll[maxLLIndex] maxLLIndex = myLLIndex; end

            #logprob = log( sum  a/sqrt(2pi)sigma * exp(-(x-mean)^2/2.0sigma^2) )
            #d logprob/da = 1/prob * 1/sqrt(2pi)sigma * exp(...)

            #d logprob/dsigma = 1/prob * 1/sqrt(2pi)sigma * exp(...) * ((x-mean)^2 /sigma^4 - 1/sigma^2)
            #d logprob/dmean = 1/prob * 1/sqrt(2pi)sigma^3*(x-m)*exp(...)

            #if log(p_max) < -500
            #d logprob/da_max = 1/a_max

            #d logprob/dsigma_max = -sqrt(2pi)+(x-mean)^2/sigma^3 + sum(p * sqrt(2pi) [1 - (x-mean)^2/sigma^2] exp(+...))
            #but: sqrt(2pi)[1-(x-mean)^2/sigma^2]exp(+...) = sqrt(2pi)sigma*exp(+...)[1/sigma-(x-mean)^2/sigma^3]
            #so d logprob/dsigma_max = -log(a*sqrt(2pi))+(x-mean)^2/sigma^3 + sum (p/p_max * [1/sigma - (x-mean)^2/sigma^3])
            #d logprob/dmean_max = (x-mean)/sigma^2 + sum(p/p_max*(x-mean)/sigma^2)

            #d logprob/da = p/(p_max*a)
            #d logprob/dsigma = a((x-m)^2/sqrt(2pi)sigma^4 - 1/sqrt(2pi)sigma^2)exp(...)/p_max
            #                 = a/sqrt(2pi)sigma*exp(...) * [(x-m)^2/sigma^3 - 1/sigma] /p_max
            #d logprob/dmean = a(x-mean)/sqrt(2pi)sigma^3*exp(...)/p_max
         end
         if ll[maxLLIndex] < -500.0
            println(event)
            lh+=ll[maxLLIndex]
            distance = event - p[maxLLIndex*3-1]
            sigma_distance_21 = distance/p[maxLLIndex*3]^2
            sigma_distance_32 = distance * sigma_distance_21 / p[maxLLIndex*3]

            gradient[maxLLIndex*3-2] = 1.0/p[maxLLIndex*3-2]
            gradient[maxLLIndex*3-1] = 1.
            gradient[maxLLIndex*3] = 0.
            #gradient[maxLLIndex*3]   =
            for i in filter(j->j!=maxLLIndex, eachindex(ll))
               prob_ratio =  exp(ll[i] - ll[maxLLIndex])
               lh += prob_ratio
               gradient[i*3-2] = prob_ratio/p[i*3-2]
               gradient[i*3-1] = prob_ratio * (event-p[i*3-1])/p[i*3]^2
               gradient[i*3] = prob_ratio*(((event-p[i*3-1])/p[i*3])^2 - 1.0)/p[i*3]
               gradient[maxLLIndex*3] += prob_ratio
               gradient[maxLLIndex*3-1]+= prob_ratio
            end
            gradient[maxLLIndex*3-1] *= sigma_distance_21
            gradient[maxLLIndex*3] *= (1/p[maxLLIndex*3] - sigma_distance_32)
            gradient[maxLLIndex*3] += -1/p[maxLLIndex*3] + sigma_distance_32
            #d log(a/sqrt(2pi)sigma) /dsigma = -d log sigma /dsigma = -1/sigma
            for i in eachindex(grad)
               grad[i] += gradient[i]
            end
         else
            #lh+= log(sum(map(logprob->exp(logprob),ll)) )
            event_likelihood = 0.
            #d logprob/da = 1/prob * 1/sqrt(2pi)sigma * exp(...)
            #d logprob/dsigma = 1/prob * 1/sqrt(2pi)sigma * exp(...) * ((x-mean)^2 /sigma^4 - 1/sigma^2)
            #d logprob/dmean = 1/prob * 1/sqrt(2pi)sigma^3*(x-m)*exp(...)
            for i in eachindex(ll)
               partial_likelihood = exp(ll[i])
               distance = event-p[i*3-1]
               inverse_sigma = 1/p[i*3]
               event_likelihood += partial_likelihood
               gradient[i*3-2] = partial_likelihood / p[3*i-2]
               gradient[i*3-1] = partial_likelihood * distance * inverse_sigma^2
               gradient[i*3]   = partial_likelihood * ((distance*inverse_sigma)^2 - 1.)*inverse_sigma
            end
            lh+=log(event_likelihood)
            gradient ./=event_likelihood

            for i in eachindex(gradient)
               grad[i] += gradient[i]
            end
         end
      end
   else
      for event in x
         ll = Array{Float64,1}(undef,Int64(length(p)/3))
         maxLLIndex = 1
         for i in 1:3:length(p)
            #amplitude = p[i]
            distance = p[i+1] - event
            sigma = p[i+2]
            norm = p[i]/sqrt(2π)/sigma
            myLLIndex = Int64( (i-1)/3 + 1 )
            ll[myLLIndex] = log(norm) - 0.5(distance/sigma)^2
            if ll[myLLIndex] > ll[maxLLIndex] maxLLIndex = myLLIndex; end
         end

         if ll[maxLLIndex] < -500.0
            #logprob ~= log(p_max) + sum (p/p_max)
            println(event)
            lh+=ll[maxLLIndex]
            for i in filter(j->j!=maxLLIndex, eachindex(ll))
               lh += exp(ll[i] - ll[maxLLIndex])
            end
   #         println(event, " ", lh)
         else
            lh+= log(sum(map(logprob->exp(logprob),ll)) )
         end
      end
   end
   return lh
end

function unitaryConstraint(p::Vector, grad::AbstractVector)
   #println("p = " ,p)
   if length(grad)>0
      for i in 1:length(grad)
         grad[i] = Float64(mod(i,3)==1)
      end
      #println("grad = " ,grad)
   end
   sum(p[1:3:9]) - 1.0
   #p[1]+p[4]+p[7] - 1.0
end


lowerBounds = [0,-Inf,0,0,-Inf,0,0,-Inf,0]
upperBounds = [1, Inf, Inf,1, Inf, Inf,1, Inf, Inf]

#NLOpt
opt = Opt(:LN_COBYLA, 9)
lower_bounds!(opt, lowerBounds)
upper_bounds!(opt, upperBounds)

max_objective!(opt, (x,g)->LogLikelihoodGaussianMixtureEstimate(x,g, z))
equality_constraint!(opt, unitaryConstraint, 1e-8)
ftol_rel!(opt, 1e-7)

(minf,minx,ret) = optimize(opt, rand(9)/3.0)
#(minf,minx,ret) = optimize(opt, minx)
@show((minf,minx))

#histogram(z, nbins = 100, normalize = true)
plot!(x->exp(LogLikelihoodGaussianMixtureEstimate(p_true, [], [x])), -10., 10.)
plot!(x->exp(LogLikelihoodGaussianMixtureEstimate(minx, [], [x])), -10., 10.)

plot(x->(LogLikelihoodGaussianMixtureEstimate(p_true, [], [x])), -100., 10.)

#test continuity
plot(x->begin exp(LogLikelihoodGaussianMixtureEstimate(p_true, b, [x]));b[9]; end , -76., -70.)
#test gradient at z[1] by parameters (not x)

using Calculus
g = Calculus.gradient(p->LogLikelihoodGaussianMixtureEstimate(p, [], [z[1] ]))
@show(g(p_true))
b_finitediff = g(p_true)
b = zeros(9)
LogLikelihoodGaussianMixtureEstimate(p_true, b, [z[1]])
@show(b)
@show(sqrt(sum((b-b_finitediff)'*(b-b_finitediff))))
