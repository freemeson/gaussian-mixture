using Random
using Plots
using Images
#using QuartzImageIO #need to be deployed
using Distributions
using SpecialFunctions
using LinearAlgebra
using Calculus
using NLopt
using Printf
using JLD2

Plots.gr()

#Random.seed!(201910222054)
Random.seed!(201920261032)
d3 = MixtureModel(MvNormal[
    MultivariateNormal([1.1, 4.0], [1.0 0.0; 0.0 2.0]),
    MultivariateNormal([2.1, 2.0], [3.0 0.0; 0.0 0.4]),
    MultivariateNormal([7.5, 5.5], [1.0 0.0; 0.0 1.0])
    ], [0.1, 0.8, 0.1])

    # push!(m3,   diagonalGauss([0.1, 1.1, 4.0, 1.0, 2.0]),
    #             diagonalGauss([0.8, 2.1, 2.0, 3.0, 0.4]),
    #             diagonalGauss([0.1, 7.5, 5.5, 1.0, 1.0]))

z3 = rand(d3, 10000)

dScale = 1.0

@enum ProbabilityParameterisation AngleAmplitude SqrtAmplitude OldSqrtAmplitude

#scatter([(i[1],i[2]) for i in eachcol(z3)])

# mutable struct diagonalGauss{T<:Real}
#     amplitude::T
#     means::Vector{T}
#     sigmas::Vector{T}
#     sigmaSigns::Vector{T}
#     function diagonalGauss(p::Array{T,1}) where T<:Real
#         mod(length(p)-1, 2) == 0 || error("length(p-1) must be divisible by 2")
#         dg = new()
#         dg.amplitude = p[1]
#         nDim = Int64((length(p)-1)/2)
#         dg.means = p[2:nDim+1]
#         dg.sigmas = 1 ./ abs.(p[nDim+2:end])
#         dg.sigmaSigns = sign.(p[nDim+2:end])
#         return dg
#     end
# end

mutable struct diagonalGauss{T<:Real}
    amplitude::T
    means::Vector{T}
    sigmas::Vector{T}
    sigmaSigns::Vector{T}
    function diagonalGauss(p::Array{T,1}) where T<:Real
        mod(length(p)-1, 2) == 0 || error("length(p-1) must be divisible by 2")

        nDim = Int64((length(p)-1)/2)
        sigmas = 1 ./ abs.(p[nDim+2:end])
        sigmaSigns = sign.(p[nDim+2:end])

        dg = new{T}(p[1], p[2:nDim+1], sigmas, sigmaSigns)

        return dg
    end
end

function probability(dg::diagonalGauss{Float64}, x::Vector)
    distance = [d*d for d in (x-dg.means) .* dg.sigmas]
    exponent = -0.5 * sum(distance)
    prob = dg.amplitude^2 * exp(exponent) * prod(dg.sigmas) / sqrt(2*π)^length(x)  # 1./|sigma| is stored in dg
    #prob = exp(dg.amplitude) * exp(exponent) * prod(dg.sigmas) / sqrt(2*π)  # 1./|sigma| is stored in dg
    #should be sqrt(2pi)^nDim
    return prob
end

function probability(dgA::Array{diagonalGauss{Float64}, 1}, x::Vector)
    sqrAmplitudeNorm = sum(dg->dg.amplitude^2, dgA)
    return sum(dg->probability(dg, x), dgA)/sqrAmplitudeNorm
end

#contour(0:0.1:5, 0:0.1:5, (x,y)->log(probability(a, [x,y])+probability(b, [x,y])), fill = true)

function dgCoordinate(dgA::Array{diagonalGauss,1}, x::Vector)
    nominators = zero(x)
    denominators = zero(x)
    # for dg in dgA
    #     amplNorm += dg.amplitude
    # end
    for dg in dgA
        distance = (x-dg.means) .* dg.sigmas  # 1./|sigma| is stored in dg
        distance2 = [d*d for d in distance]
        exponents = -0.5*distance2
        gaussNorms = dg.sigmas / sqrt(2*π)
        gausses = exp.(exponents) .* gaussNorms
        cdfs = 0.5*erf.(distance / sqrt(2)) .+ 0.5
#first attempt
        prodGausses = prod(gausses)
        #println(prodGausses)
        denominatorParts = dg.amplitude * prodGausses ./ gausses
        nominatorsParts = denominatorParts .* cdfs

        nominators += nominatorsParts
        denominators += denominatorParts
    end
    return nominators ./ denominators
end

function dgMarginalMean(dgA::Array{diagonalGauss{Float64},1}, x::Vector)
    nDim = length(x)
    m = zeros(length(dgA[1].means)-nDim)
    w = 0.0
    for dg in dgA
        prob = dg.amplitude^2 * prod([exp(-0.5d*d)/sqrt(2π) for d in (x - dg.means[1:nDim]).*dg.sigmas[1:nDim] ])*prod(dg.sigmas[1:nDim])
        m .+= prob*dg.means[nDim+1:end]
        w += prob
    end

    return m/w

end

function dgMarginalProbability(dgA::Array{diagonalGauss{Float64},1}, x::Vector)
    nDim = length(x)
    jointProbabilities = zeros(nDim)
    for dg in dgA
        distance = [d*d for d in (x - dg.means) .* dg.sigmas]
        exponents = -0.5*distance
        gaussNorms = dg.sigmas / sqrt(2π)
#        gausses = exp.(exponents) .* gaussNorms
#        cdfs = 0.5*erf.(distance/sqrt(2)) .+ 0.5
        gaussProd = 0.0
        gaussTriangleProd = zeros(nDim)
        for nu in 1:nDim
            gaussProd+= exponents[nu] +log(gaussNorms[nu])
            gaussTriangleProd[nu] = gaussProd
        end
        gaussTriangleProd = exp.(gaussTriangleProd)
        jointProbabilities += dg.amplitude * gaussTriangleProd
    end
    marginalProbabilities = deepcopy(jointProbabilities)
    for nu in 2:nDim
        marginalProbabilities[nu] /= jointProbabilities[nu-1]
    end
    return marginalProbabilities
end

function dgMarginalCDF(dgA::Array{diagonalGauss{R},1}, x::Vector) where R<:Real
    nDim = length(x)
    jointProbabilities = zeros(nDim)
    marginalCDFs = zeros(nDim)
    #println(x)
    for dg in dgA
        distance = (x - dg.means) .* dg.sigmas
        distance2 = [d*d for d in distance ]
        exponents = -0.5*distance2
        gaussNorms = dg.sigmas / sqrt(2π)
#        gausses = exp.(exponents) .* gaussNorms
        cdfs = 0.5*erf.(distance/sqrt(2)) #.+ 0.
        #println(cdfs...)
        gaussProd = 0.0
        gaussTriangleProd = zeros(nDim)
        for nu in 1:nDim # the last one is unnecessary
            gaussProd+= exponents[nu] +log(gaussNorms[nu])
            gaussTriangleProd[nu] = gaussProd
        end
        gaussTriangleProd = exp.(gaussTriangleProd)
        gaussTriangleCDFProd = zeros(nDim)
        gaussTriangleCDFProd[1] = cdfs[1]
        for nu in 2:nDim
            gaussTriangleCDFProd[nu] = cdfs[nu] * gaussTriangleProd[nu-1]
        end
        jointProbabilities += dg.amplitude^2 * gaussTriangleProd
        marginalCDFs += dg.amplitude^2 * gaussTriangleCDFProd
    end

    for nu in 2:nDim
        marginalCDFs[nu] /= jointProbabilities[nu-1]
    end
    return marginalCDFs
end

function plotMarginalCDFV3(p::Vector, pl = plot(),plotRange = -10:0.1:10)
    dgParamSize=3
    nGauss = div(length(p)+2,2*dgParamSize)
    dgAParamSize = div(length(p),2)
    dgA = dgAConstructSwitch(p[1:dgAParamSize],1,AngleAmplitude)
    plot!(pl, x->dgMarginalCDF(dgA,[x])[1], plotRange)
end

# function dgMarginalCDFDerivative(dgA::Array{diagonalGauss{R},1},
#     x::Vector{R},
#     dFnu_dampl::Union{Nothing, Array{R, 2}} = nothing,
#     dFnu_dmean::Union{Nothing, Array{R, 3}} = nothing,
#     dFnu_dsigm::Union{Nothing, Array{R, 3}} = nothing, checkSize = false) where R<:Real
#
#     nDim = length(x)
#     nGauss = length(dgA)
#
#     if typeof(dFnu_dampl)!=Nothing && typeof(dFnu_dmean)!=Nothing && typeof(dFnu_dsigm)!=Nothing
#         argumentOutput = true
#         if checkSize
#             size(x) == nDim || error("size(x) == nDim")
#             size(dFnu_dampl) == (nGauss, nDim) || error("size(dFnu_dampl) == (nGauss, nDim)")
#             size(dFnu_dmean) == (nGauss, nDim, nDim) || error("size(dFnu_dmean) == (nGauss, nDim, nDim)")
#             size(dFnu_dsigm) == (nGauss, nDim, nDim) || error("size(dFnu_dsigm) == (nGauss, nDim, nDim)")
#         end
#     end

function amplitudeToProbability(amplitudes::Vector{R}, probabilities::Vector{R}, gradientMatrix::Array{R,2}) where R<:Real

    probabilities[:] = [x^2 for x in amplitudes]
    sumAmplitudes = sum(probabilities)
    probabilities[:] *= 1.0/sumAmplitudes
    #println(probabilities)

    for i in 1:length(amplitudes)
        for j in 1:length(amplitudes)
            gradientMatrix[j,i] = -2*amplitudes[i]^2*amplitudes[j]/sumAmplitudes^2
        end
        gradientMatrix[i,i] += 2*amplitudes[i]/sumAmplitudes
    end
end

function sqrtAmplitudeToAngles(amplitudes::Vector{R}, sqrtNormalize = true) where R<:Real
    sqrtNorm = 1.0
    if sqrtNormalize
        sqrtNorm = sqrt(sum(x->x*x, amplitudes))
        #println(sqrtNorm^2)
    end
    angles = zeros(length(amplitudes)-1)
    cosAlphaProd = 1.0
    for i in 1:length(angles)
        angles[i] = asin(amplitudes[i+1]/sqrtNorm)/cosAlphaProd
        cosAlphaProd *= cos(angles[i])
    end
    return angles
end

function angleToProbabilityAmplitudeUnpack(angles::Vector{R}, proabilities::Vector{R}, gradientMatrix::Array{R,2} ) where R<:Real
    cosAlphaProd = 1.0

    for i in 1:length(angles)
        proabilities[i+1] = (cosAlphaProd*sin(angles[i]))^2
        cosAlphaProd *= cos(angles[i])
    end

    cosAlpha = [cos(alpha) for alpha in angles]
    sinAlpha = [sin(alpha) for alpha in angles]
    sinCosAlpha = 2.0*sinAlpha .* cosAlpha

    proabilities[1] = cosAlphaProd^2
    for i in 1:length(proabilities)
        for j in 1:length(angles)
            trigProduct = -1.0
            if i==1
                for k in 1:length(angles)
                    if k!=j
                        trigProduct *= cosAlpha[k]^2
                    else
                        trigProduct *= sinCosAlpha[k]
                    end
                end
                gradientMatrix[j,i] = trigProduct
            else
                if j+1>i
                    gradientMatrix[j,i] = 0.0
                else
                #gradientMatrix[i,j] = 1.0
                    trigProduct = 1.0
                    if j+1==i
                        trigProduct *= sinCosAlpha[j]
                        for k in 1:j-1
                            trigProduct *= cosAlpha[k]^2
                        end
                    else
                        trigProduct *= sinAlpha[i-1]^2
                        for k in 1:i-2
                            if k!=j
                                trigProduct *= cosAlpha[k]^2
                            else
                                trigProduct *= - sinCosAlpha[k]
                            end
                            #println(k)
                        end
                        #trigProduct = -99.0

                    end
                    gradientMatrix[j,i] = trigProduct
                end
            end
        end
    end

    #amplitudes = amplitudes .* amplitudes
end

mutable struct marginalCDFDerivativeAllocations{T<:Real}
    n1::Array{T,2}
    n2::Vector{T}
    n3::Array{T,2}
    n4::Array{T,2}
    n5::Array{T,2}
    n6::Array{T,2}
    d1::Vector{T}

    distance::Vector{T}
    distance2::Vector{T}
    exponents::Vector{T}
    gaussNorms::Vector{T}
    cdfs::Vector{T}
    gaussTriangleProd::Vector{T}

    fampl::Array{T,2}

    dFdm::Array{T,3}
    dFds::Array{T,3}
    dFda::Array{T,2}
    dFdx::Array{T,2}
    dFdx_inv::Array{T,2}
    logProbabilities::Vector{T}
    Da::Vector{T}
    Dm::Array{T,2}
    Ds::Array{T,2}
    probParams::Vector{T}
    normalizedAmplitudes::Vector{T}
    amplitudeGradient::Array{T,2}
    perturbationVec::Vector{T}
    perturbationSumLog::T

    function marginalCDFDerivativeAllocations{T}(nDim::Int64, nGauss::Int64, nProbParams::Int64) where T<:Real
        nProbParamsPlaceholder = nProbParams == 0 ? 1 : nProbParams
        mA = new{T}(
            zeros(nGauss, nDim),
            zeros(nDim),
            zeros(nGauss, nDim),
            zeros(nGauss, nDim),
            zeros(nGauss, nDim),
            zeros(nGauss, nDim),
            zeros(nDim),
            zeros(nDim),
            zeros(nDim),
            zeros(nDim),
            zeros(nDim),
            zeros(nDim),
            zeros(nDim),
            zeros(nGauss, nDim),

            zeros(nGauss, nDim, nDim),
            zeros(nGauss, nDim, nDim),
            zeros(nProbParamsPlaceholder, nDim),
            zeros(nDim, nDim),
            zeros(nDim, nDim),
            zeros(nGauss),
            zeros(nProbParamsPlaceholder),
            zeros(nGauss,nDim),
            zeros(nGauss,nDim),
            zeros(nProbParamsPlaceholder),
            zeros(nGauss),
            zeros(nProbParamsPlaceholder,nGauss),
            zeros(nDim),
            0
        )
        return mA
    end
end

function resetMarginalCDFAllocations(mA::marginalCDFDerivativeAllocations{T}) where T<:Real
    mA.n1 .= 0
    mA.n2 .= 0
    mA.n3 .= 0
    mA.n4 .= 0
    mA.n5 .= 0
    mA.n6 .= 0
    mA.d1 .= 0
    mA.fampl .= 0
    mA.dFdx .= 0
    mA.dFdm .= 0
    mA.dFds .= 0
    mA.dFda .= 0
    mA.perturbationVec .= 0
    mA.logProbabilities .= 0
    #perturbationSumLog is anulled only after each data cycle

    return nothing
end

function dgExternalAmplitudesMarginalCDFDerivative(dgA::Array{diagonalGauss{R},1},
    normalizedAmplitudes::Vector{R},
    amplitudeGradient::Array{R,2},
    x::Vector{R},
    dFnu_dampl::Array{R, 2},
    dFnu_dmean::Array{R, 3},
    dFnu_dsigm::Array{R, 3},
    logProbability::Array{R, 1},
    checkSize = false) where R<:Real

    verbose = false
    nDim = length(x)
    nGauss = length(dgA)
    nAmplitudeParams = size(amplitudeGradient)[1]

    if verbose
        println("dgExternalAmplitudesMarginalCDFDerivative")
        println(normalizedAmplitudes)
        println(nAmplitudeParams)
    end
    if typeof(dFnu_dampl)!=Nothing && typeof(dFnu_dmean)!=Nothing && typeof(dFnu_dsigm)!=Nothing
        argumentOutput = true
        if checkSize
            size(x) == nDim || error("size(x) == nDim")
            size(normalizedAmplitudes) == nGauss || error("size(amplitudes) == nGauss")
#            size(amplitudeGradient)
            size(amplitudeGradient)[1] == nGauss || error("size(amplitudeGradient)[1] == nGauss")
            size(dFnu_dampl) == (nAmplitudeParams, nDim) || error("size(dFnu_dampl) == (nAmplitudeParams, nDim)")
            size(dFnu_dmean) == (nGauss, nDim, nDim) || error("size(dFnu_dmean) == (nGauss, nDim, nDim)")
            size(dFnu_dsigm) == (nGauss, nDim, nDim) || error("size(dFnu_dsigm) == (nGauss, nDim, nDim)")
            size(logProbability) == (nGauss) || error("size(logProbability) == (nGauss)")
        end
    end

    n1 = zeros(nGauss, nDim) #erf * triangleProd, for each diagonalGauss and dimension
    n2 = zeros(nDim) #sum of amplitude * erf* triangleProd, for each dimension, equals joint probability
    n3 = zeros(nGauss, nDim) #triangleProd for each diagonalGauss and dimension
    n4 = zeros(nGauss, nDim) # ampl*distance*sign(sigma)/abs(sigma)

    n5 = zeros(nGauss, nDim) #(x-mean)/sigma^2
    n6 = zeros(nGauss, nDim) #(x-mean)^2/sigma^2  /sigma
    d1 = zeros(nDim) #sum of amplitude*triangleProd, needed for marginal probability


    for (k,dg) in enumerate(dgA)
        distance = (x - dg.means) .* dg.sigmas
        distance2 = [d*d for d in distance ]
        exponents = -0.5*distance2
        gaussNorms = dg.sigmas / sqrt(2π)
#        gausses = exp.(exponents) .* gaussNorms
        cdfs = 0.5*erf.(distance/sqrt(2))# .+ 0.5
        gaussProd = 0.0
        gaussTriangleProd = zeros(nDim)
        for nu in 1:nDim
            gaussProd+= exponents[nu] + log(gaussNorms[nu])
            gaussTriangleProd[nu] = gaussProd
        end
        gaussTriangleProd = exp.(gaussTriangleProd)
        logProbability[k] = gaussProd + log(normalizedAmplitudes[k])

        d1 += normalizedAmplitudes[k]*gaussTriangleProd

        n1[k,1] = cdfs[1]
        n3[k,1] = gaussTriangleProd[1]
        for nu in 2:nDim
            n1[k,nu] = cdfs[nu] * gaussTriangleProd[nu-1]
            n3[k,nu] = gaussTriangleProd[nu] #last triangleProd is not needed yet
        end
        for nu in 1:nDim
            n2[nu] += normalizedAmplitudes[k]*n1[k, nu]
            n4[k,nu] = normalizedAmplitudes[k]*distance[nu]*dg.sigmaSigns[nu] #distance[nu] contains a sigma
            n5[k,nu] = distance[nu] * dg.sigmas[nu]
            n6[k,nu] = distance2[nu] * dg.sigmas[nu] * dg.sigmaSigns[nu] - dg.sigmas[nu]*dg.sigmaSigns[nu]
        end
    end



    fampl = zeros(nGauss, nDim)
    for k in 1:nGauss
        fampl[k, 1] = n1[k, 1]
        for nu in 2:nDim
            fampl[k, nu] = n1[k,nu]/d1[nu-1] - n2[nu] * n3[k,nu-1] /d1[nu-1]^2
        end
    end



    # @show(normalizedAmplitudes)
    # @show(sumAmpl)
    # @show(dAmplNorm)
    # @show(fampl)
    #return dAmplNorm
    for k in 1:nGauss, nu in 1:nDim,lambda in 1:nDim
        dFnu_dmean[k, nu, lambda] = 0.0
        dFnu_dsigm[k, nu, lambda] = 0.0
    end

    for k in 1:nAmplitudeParams, nu in 1:nDim
        dFnu_dampl[k, nu] = 0.0
    end
    for k in 1:nAmplitudeParams, nu in 1:nDim
        for l in 1:nGauss
            dFnu_dampl[k,nu] += amplitudeGradient[k,l] * fampl[l,nu]
        end
    end

    if verbose
        println("   amplitudeGradient:")
        println(amplitudeGradient)
        println("    fampl")
        println(fampl)
        println("    dFnu_dampl")
        println(dFnu_dampl)
    end

    for k in 1:nGauss
        dFnu_dmean[k,1,1] = -normalizedAmplitudes[k]* n3[k,1]
        dFnu_dsigm[k,1,1] = -n4[k, 1]*n3[k,1]

        for nu in 2:nDim
            dFnu_dmean[k, nu, nu] = -normalizedAmplitudes[k]*n3[k,nu]/d1[nu-1]
            dFnu_dsigm[k, nu, nu] = -n4[k, nu]*n3[k,nu]/d1[nu-1]
            for lambda in 1:nu-1
                commonTerm = normalizedAmplitudes[k]*(n1[k, nu]/d1[nu-1] - n2[nu] * n3[k, nu-1]/d1[nu-1]^2)
                dFnu_dmean[k, nu, lambda] = n5[k,lambda]*commonTerm
                dFnu_dsigm[k, nu, lambda] = n6[k,lambda]*commonTerm
            end
        end


    end

end

function testGauss(x::R, s::R) where R<:Real
    a::R = exp(-((x/s)^2)/2)/sqrt(2π*s^2)
    b::R = -0.5*log(2π*s^2) -((x/s)^2/2)
    println(log(a))
    println(b)
    return log(a) - b
end

@polly @inbounds function dgExternalAllocationsAmplitudesMarginalCDFDerivative(dgA::Array{diagonalGauss{R},1},
    # normalizedAmplitudes::Vector{R},
    # amplitudeGradient::Array{R,2},
    x::Vector{R},
    # dFnu_dampl::Array{R, 2},
    # dFnu_dmean::Array{R, 3},
    # dFnu_dsigm::Array{R, 3},
    # logProbability::Array{R, 1},
    mA::marginalCDFDerivativeAllocations{R}) where R<:Real

    verbose = false
    nDim = length(x)
    nGauss = length(dgA)
    nAmplitudeParams = size(mA.amplitudeGradient)[1]

    if verbose
        println("dgExternalAmplitudesMarginalCDFDerivative")
        println(mA.normalizedAmplitudes)
        println(mA.nAmplitudeParams)
    end
#     if typeof(dFnu_dampl)!=Nothing && typeof(dFnu_dmean)!=Nothing && typeof(dFnu_dsigm)!=Nothing
#         argumentOutput = true
#         if checkSize
#             size(x) == nDim || error("size(x) == nDim")
#             size(normalizedAmplitudes) == nGauss || error("size(amplitudes) == nGauss")
# #            size(amplitudeGradient)
#             size(amplitudeGradient)[1] == nGauss || error("size(amplitudeGradient)[1] == nGauss")
#             size(dFnu_dampl) == (nAmplitudeParams, nDim) || error("size(dFnu_dampl) == (nAmplitudeParams, nDim)")
#             size(dFnu_dmean) == (nGauss, nDim, nDim) || error("size(dFnu_dmean) == (nGauss, nDim, nDim)")
#             size(dFnu_dsigm) == (nGauss, nDim, nDim) || error("size(dFnu_dsigm) == (nGauss, nDim, nDim)")
#             size(logProbability) == (nGauss) || error("size(logProbability) == (nGauss)")
#         end
#     end

    resetMarginalCDFAllocations(mA)
    # mA.n1 = zeros(nGauss, nDim) #erf * triangleProd, for each diagonalGauss and dimension
    # mA.n2 = zeros(nDim) #sum of amplitude * erf* triangleProd, for each dimension, equals joint probability
    # mA.n3 = zeros(nGauss, nDim) #triangleProd for each diagonalGauss and dimension
    # mA.n4 = zeros(nGauss, nDim) # ampl*distance*sign(sigma)/abs(sigma)
    #
    # mA.n5 = zeros(nGauss, nDim) #(x-mean)/sigma^2
    # mA.n6 = zeros(nGauss, nDim) #(x-mean)^2/sigma^2  /sigma
    # mA.d1 = zeros(nDim) #sum of amplitude*triangleProd, needed for marginal probability


    for (k,dg) in enumerate(dgA)
#        mA.distance .= (x - dg.means) .* dg.sigmas
        @. mA.distance = (x-dg.means) * dg.sigmas
        #mA.distance2 .= [d*d for d in mA.distance ]
        @. mA.distance2 = mA.distance^2
        #mA.exponents .= -0.5*mA.distance2
        @. mA.exponents = -0.5*mA.distance2
        #mA.gaussNorms .= dg.sigmas / sqrt(2π)
        @. mA.gaussNorms = dg.sigmas / sqrt(2π)
#        gausses = exp.(exponents) .* gaussNorms
        #mA.cdfs .= 0.5*erf.(mA.distance/sqrt(2))# .+ 0.5
        @. mA.cdfs = 0.5*erf(mA.distance/sqrt(2))# .+ 0.5
        gaussProd = 0.0
        #gaussTriangleProd .= 0
        for nu in 1:nDim
        @inbounds    gaussProd+= mA.exponents[nu] + log(mA.gaussNorms[nu])
        @inbounds    mA.gaussTriangleProd[nu] = gaussProd
        end
        #mA.gaussTriangleProd .= exp.(mA.gaussTriangleProd)
        @. mA.gaussTriangleProd = exp(mA.gaussTriangleProd)
        mA.logProbabilities[k] = gaussProd + log(mA.normalizedAmplitudes[k])
        #mA.d1 .+= mA.normalizedAmplitudes[k]*mA.gaussTriangleProd
        @. mA.d1 += mA.normalizedAmplitudes[k]*mA.gaussTriangleProd

        @inbounds mA.n1[k,1] = mA.cdfs[1]
        @inbounds mA.n3[k,1] = mA.gaussTriangleProd[1]
        for nu in 2:nDim
        @inbounds    mA.n1[k,nu] = mA.cdfs[nu] * mA.gaussTriangleProd[nu-1]
        @inbounds    mA.n3[k,nu] = mA.gaussTriangleProd[nu] #last triangleProd is not needed yet
        end
        # for nu in 1:nDim
        #     mA.n2[nu] += mA.normalizedAmplitudes[k]*mA.n1[k, nu]
        #     mA.n4[k,nu] = mA.normalizedAmplitudes[k]*mA.distance[nu]*dg.sigmaSigns[nu] #distance[nu] contains a sigma
        #     mA.n5[k,nu] = mA.distance[nu] * dg.sigmas[nu]
        #     mA.n6[k,nu] = mA.distance2[nu] * dg.sigmas[nu] * dg.sigmaSigns[nu] - dg.sigmas[nu]*dg.sigmaSigns[nu]
        # end
        @. mA.n2 += mA.normalizedAmplitudes[k]*mA.n1[k, :]
        @. mA.n4[k,:] = mA.normalizedAmplitudes[k]*mA.distance*dg.sigmaSigns
        @. mA.n5[k,:] = mA.distance * dg.sigmas #distance[nu] contains a sigma
        @. mA.n6[k,:] = mA.distance2 * dg.sigmas * dg.sigmaSigns - dg.sigmas*dg.sigmaSigns
    end



    #mA.fampl = zeros(nGauss, nDim)
    for k in 1:nGauss
        @inbounds mA.fampl[k, 1] = mA.n1[k, 1]
        for nu in 2:nDim
        @inbounds    mA.fampl[k, nu] = mA.n1[k,nu]/mA.d1[nu-1] - mA.n2[nu] * mA.n3[k,nu-1] /mA.d1[nu-1]^2
        end
    end



    # @show(normalizedAmplitudes)
    # @show(sumAmpl)
    # @show(dAmplNorm)
    # @show(fampl)
    #return dAmplNorm
    # for k in 1:nGauss, nu in 1:nDim,lambda in 1:nDim
    #     dFnu_dmean[k, nu, lambda] = 0.0
    #     dFnu_dsigm[k, nu, lambda] = 0.0
    # end
    #
    # for k in 1:nAmplitudeParams, nu in 1:nDim
    #     dFnu_dampl[k, nu] = 0.0
    # end
    for k in 1:nAmplitudeParams, nu in 1:nDim
        for l in 1:nGauss
        @inbounds    mA.dFda[k,nu] += mA.amplitudeGradient[k,l] * mA.fampl[l,nu]
        end
    end

    if verbose
        println("   amplitudeGradient:")
        println(mA.amplitudeGradient)
        println("    mA.fampl")
        println(mA.fampl)
        println("    dFnu_dampl")
        println(mA.dFda)
    end

    for k in 1:nGauss
        @inbounds mA.dFdm[k,1,1] = -mA.normalizedAmplitudes[k]* mA.n3[k,1]
        @inbounds mA.dFds[k,1,1] = -mA.n4[k, 1]*mA.n3[k,1]

        for nu in 2:nDim
            @inbounds mA.dFdm[k, nu, nu] = -mA.normalizedAmplitudes[k]*mA.n3[k,nu]/mA.d1[nu-1]
            @inbounds mA.dFds[k, nu, nu] = -mA.n4[k, nu]*mA.n3[k,nu]/mA.d1[nu-1]
            for lambda in 1:nu-1
                @inbounds commonTerm = mA.normalizedAmplitudes[k]*(mA.n1[k, nu]/mA.d1[nu-1] - mA.n2[nu] * mA.n3[k, nu-1]/mA.d1[nu-1]^2)
                @inbounds mA.dFdm[k, nu, lambda] = mA.n5[k,lambda]*commonTerm
                @inbounds mA.dFds[k, nu, lambda] = mA.n6[k,lambda]*commonTerm
            end
        end


    end
    return nothing
end

function dgMarginalCDFDerivative(dgA::Array{diagonalGauss{R},1},
    x::Vector{R},
    dFnu_dampl::Array{R, 2},
    dFnu_dmean::Array{R, 3},
    dFnu_dsigm::Array{R, 3},
    logProbability::Array{R, 1}, checkSize = false) where R<:Real

    nDim = length(x)
    nGauss = length(dgA)

    if typeof(dFnu_dampl)!=Nothing && typeof(dFnu_dmean)!=Nothing && typeof(dFnu_dsigm)!=Nothing
        argumentOutput = true
        if checkSize
            size(x) == nDim || error("size(x) == nDim")
            size(dFnu_dampl) == (nGauss, nDim) || error("size(dFnu_dampl) == (nGauss, nDim)")
            size(dFnu_dmean) == (nGauss, nDim, nDim) || error("size(dFnu_dmean) == (nGauss, nDim, nDim)")
            size(dFnu_dsigm) == (nGauss, nDim, nDim) || error("size(dFnu_dsigm) == (nGauss, nDim, nDim)")
            size(logProbability) == (nGauss) || error("size(logProbability) == (nGauss)")
        end
    end

    n1 = zeros(nGauss, nDim) #erf * triangleProd, for each diagonalGauss and dimension
    n2 = zeros(nDim) #sum of amplitude * erf* triangleProd, for each dimension, equals joint probability
    n3 = zeros(nGauss, nDim) #triangleProd for each diagonalGauss and dimension
    n4 = zeros(nGauss, nDim) # ampl*distance*sign(sigma)/abs(sigma)

    n5 = zeros(nGauss, nDim) #(x-mean)/sigma^2
    n6 = zeros(nGauss, nDim) #(x-mean)^2/sigma^2  /sigma
    d1 = zeros(nDim) #sum of amplitude*triangleProd, needed for marginal probability

    sumAmpl = 0.0
    normalizedAmplitudes = zeros(nGauss)
    for k in 1:nGauss

#        sumAmpl += dgA[k].amplitude
        sumAmpl += dgA[k].amplitude^2
        # amp = exp(dgA[k].amplitude)
        # sumAmpl += amp
        # normalizedAmplitudes[k] = amp
    end

    for k in 1:nGauss
#        normalizedAmplitudes[k] = dgA[k].amplitude / sumAmpl
        normalizedAmplitudes[k] = dgA[k].amplitude^2 / sumAmpl
#        normalizedAmplitudes[k] /= sumAmpl
    end

    println("dgMarginalCDFDerivative")
    println(normalizedAmplitudes)

    for (k,dg) in enumerate(dgA)
        distance = (x - dg.means) .* dg.sigmas
        distance2 = [d*d for d in distance ]
        exponents = -0.5*distance2
        gaussNorms = dg.sigmas / sqrt(2π)
#        gausses = exp.(exponents) .* gaussNorms
        cdfs = 0.5*erf.(distance/sqrt(2))# .+ 0.5
        gaussProd = 0.0
        gaussTriangleProd = zeros(nDim)
        for nu in 1:nDim
            gaussProd+= exponents[nu] + log(gaussNorms[nu])
            gaussTriangleProd[nu] = gaussProd
        end
        gaussTriangleProd = exp.(gaussTriangleProd)
        logProbability[k] = gaussProd + log(normalizedAmplitudes[k])

        d1 += normalizedAmplitudes[k]*gaussTriangleProd

        n1[k,1] = cdfs[1]
        n3[k,1] = gaussTriangleProd[1]
        for nu in 2:nDim
            n1[k,nu] = cdfs[nu] * gaussTriangleProd[nu-1]
            n3[k,nu] = gaussTriangleProd[nu] #last triangleProd is not needed yet
        end
        for nu in 1:nDim
            n2[nu] += normalizedAmplitudes[k]*n1[k, nu]
            n4[k,nu] = normalizedAmplitudes[k]*distance[nu]*dg.sigmaSigns[nu] #distance[nu] contains a sigma
            n5[k,nu] = distance[nu] * dg.sigmas[nu]
            n6[k,nu] = distance2[nu] * dg.sigmas[nu] * dg.sigmaSigns[nu] - dg.sigmas[nu]*dg.sigmaSigns[nu]
        end
    end



    fampl = zeros(nGauss, nDim)
    for k in 1:nGauss
        fampl[k, 1] = n1[k, 1]
        for nu in 2:nDim
            fampl[k, nu] = n1[k,nu]/d1[nu-1] - n2[nu] * n3[k,nu-1] /d1[nu-1]^2
        end
    end
    dAmplNorm = fill(-1.0/sumAmpl^2,(nGauss, nGauss))
#    dAmplNorm = zeros(nGauss, nGauss)
    for k in 1:nGauss, l in 1:nGauss
#            dAmplNorm[k,l] *= dgA[l].amplitude
            dAmplNorm[k,l] *= 2*dgA[l].amplitude^2*dgA[k].amplitude
#            dAmplNorm[k,l] = -normalizedAmplitudes[k]*normalizedAmplitudes[l]
    end
    for k in 1:nGauss
#        dAmplNorm[k,k] += 1.0/sumAmpl
        dAmplNorm[k,k] += 2.0*dgA[k].amplitude/sumAmpl
#        dAmplNorm[k,k] += normalizedAmplitudes[k]
    end

    # @show(normalizedAmplitudes)
    # @show(sumAmpl)
    # @show(dAmplNorm)
    # @show(fampl)
    #return dAmplNorm
    for k in 1:nGauss, nu in 1:nDim
        dFnu_dampl[k,nu] = 0.0
        for lambda in 1:nDim
            dFnu_dmean[k, nu, lambda] = 0.0
            dFnu_dsigm[k, nu, lambda] = 0.0
        end
    end

    for k in 1:nGauss, nu in 1:nDim
        for l in 1:nGauss
            dFnu_dampl[k,nu] += dAmplNorm[k,l] * fampl[l,nu]
        end
    end

    println("   dAmplNorm:")
    println(dAmplNorm)
    println("    fampl")
    println(fampl)
    println("    dFnu_dampl")
    println(dFnu_dampl)


    for k in 1:nGauss
        dFnu_dmean[k,1,1] = -normalizedAmplitudes[k]* n3[k,1]
        dFnu_dsigm[k,1,1] = -n4[k, 1]*n3[k,1]

        for nu in 2:nDim
            dFnu_dmean[k, nu, nu] = -normalizedAmplitudes[k]*n3[k,nu]/d1[nu-1]
            dFnu_dsigm[k, nu, nu] = -n4[k, nu]*n3[k,nu]/d1[nu-1]
            for lambda in 1:nu-1
                commonTerm = normalizedAmplitudes[k]*(n1[k, nu]/d1[nu-1] - n2[nu] * n3[k, nu-1]/d1[nu-1]^2)
                dFnu_dmean[k, nu, lambda] = n5[k,lambda]*commonTerm
                dFnu_dsigm[k, nu, lambda] = n6[k,lambda]*commonTerm
            end
        end


    end

end

function dgCoordinateDiffDeterminant(dgA::Array{diagonalGauss,1}, x::Vector{Float64})
    nGauss = length(dgA)
    nDim = length(dgA[1].means)
    dFda = zeros(nGauss, nDim)
    dFdm = zeros(nGauss, nDim, nDim)
    dFds = zeros(nGauss, nDim, nDim)
    dgCoordinateDerivateNoColons(dgA, x, dFda, dFdm, dFds)
    dFdx = zeros(nDim, nDim)
    for nu in 1:nDim, lambda in 1:nDim, k in 1:nGauss
        dFdx[nu,lambda] -= dFdm[k, nu, lambda]
    end

    return det(dFdx)
    #return abs(det(dFdx))
end

function dgAConstruct(dgAv::Array{R,1}, nDim = 2) where R<:Real
    dgA = Array{diagonalGauss{R}, 1}()
    for p in Iterators.partition(dgAv, nDim*2+1)
        push!(dgA, diagonalGauss(p[1:end]))
    end
    return dgA
end

function dgAConstructSwitch(dgAv::Array{R,1},
        nDim::Int64 = 2,
        probHandling::ProbabilityParameterisation = SqrtAmplitude,
        probabilitiesOutput::Union{Vector{R},Nothing}=nothing,
        probabilityGradientOutput::Union{Array{R, 2}, Nothing} = nothing ) where R<:Real
    verbose = false
    dgA = Array{diagonalGauss{R}, 1}()
    if probHandling == SqrtAmplitude
        nGauss = div(length(dgAv) , (nDim*2+1))
        sqrtAmplitudes = dgAv[1:nGauss]
        probabilities = zeros(R,nGauss)
        probabilityGradient = zeros(R,nGauss, nGauss)
        amplitudeToProbability(sqrtAmplitudes, probabilities, probabilityGradient)
        for (a,p) in zip(sqrtAmplitudes,Iterators.partition(dgAv[nGauss+1:end], nDim*2))
            if  verbose
                println(p)
            end
            push!(dgA, diagonalGauss(vcat(a,p[1:end])))
        end
    else
        if probHandling == AngleAmplitude
            nGauss = div((length(dgAv)+1) , (nDim*2+1))
            probabilities = zeros(R,nGauss)
            if nGauss == 1
                probabilities[1] = 1.0
                probabilityGradient = zeros(R,1,1)
            else
                angles = dgAv[1:nGauss-1]
                probabilityGradient = zeros(R,nGauss-1, nGauss)

                angleToProbabilityAmplitudeUnpack(angles, probabilities, probabilityGradient)
            end
            for (a,p) in zip(probabilities,Iterators.partition(dgAv[nGauss:end], nDim*2))
                push!(dgA, diagonalGauss(vcat(sqrt(a), p[1:end])) )
            end
        end
    end
    if probabilitiesOutput != nothing && probabilityGradientOutput != nothing
        if verbose
            println("dgAConstructSwitch")
            println(probabilitiesOutput)
            println(probabilities)
        end
        probabilitiesOutput[:] = probabilities[:]
        if !(size(probabilityGradient)==(1,1) && probabilityGradient[1,1] == 0)
            probabilityGradientOutput[:,:] = probabilityGradient[:,:]
        end
    end
    return dgA
end

#dgAConstructSwitch([1.2,1.1,2.2,3.3,4.4])

function setMeans(dgA::Array{diagonalGauss,1}, means::Vector)
    nDim = length(dgA[1].means)
    for (dg,m) in zip(dgA,Iterators.partition(means,nDim))
        dg.means[:] = m[:]
    end
end

function setAmplitudes(dgA::Array{diagonalGauss,1}, amplitudes::Vector)
    for (dg,a) in zip(dgA,amplitudes)
        dg.amplitude = a
    end
end

function setSigmas(dgA::Array{diagonalGauss,1}, sigmas::Vector)
    nDim = length(dgA[1].means)
    for (dg,s) in zip(dgA, Iterators.partition(sigmas,nDim))
        dg.sigmas = 1 ./ abs.(s)
        dg.sigmaSigns = sign.(s)
    end
end

function setParameters(dgA::Array{diagonalGauss{R},1}, params::Vector{R}) where R<:Real
    nDim = length(dgA[1].means)
    chunkSize = 2*nDim+1
    for (dg,p) in zip(dgA, Iterators.partition(params, chunkSize))
        dg.amplitude = p[1]
        dg.means .= p[2:nDim+1]
        dg.sigmas .= 1.0 ./ abs.(p[nDim+2:end])
        dg.sigmaSigns .= sign.(p[nDim+2:end])
    end
end

function randomGaussStartingPoint(  nDim = 2,
                                    n = 1,
                                    meanMean = 0.0,
                                    meanLogSigma = 0.0,
                                    logSigmaMean = 0.0,
                                    logSigmaLogSigma = 0.0,
                                    minSigma = 0.0,
                                    argumentOutput = false)
    if !argumentOutput
        dgA = Array{diagonalGauss{Float64},1}()
        for k in 1:n
            push!(dgA, diagonalGauss(vcat(rand(1), rand(Normal(meanMean, exp(meanLogSigma)),nDim), minSigma .+ exp.( rand(Normal(logSigmaMean,exp(logSigmaLogSigma)),nDim)) ) ))
        end
        return dgA
    else
        dgParm = Array{Float64, 1}()
        for k in 1:n
            append!(dgParm, rand(1))
            append!(dgParm, rand(Normal(meanMean, exp(meanLogSigma)),nDim))
            append!(dgParm, minSigma .+ exp.( rand(Normal(logSigmaMean,exp(logSigmaLogSigma)),nDim)))
        end
        return dgParm
    end
end

function printGaussianParams(params::Vector, nDim::Int64 = 1, probHandling::ProbabilityParameterisation = AngleAmplitude, amplitudeCut = 1e-6, sigmaCut = 1e-6)

    dgParamSize =  nDim*2+1
    nGauss = div(length(params), 2*dgParamSize)
    dgAParamSize = nGauss*dgParamSize
    sqrAmplitudeNorm = sum(x->x^2, params[1:dgParamSize:dgAParamSize] )

    for p in Iterators.partition(params[1:dgAParamSize], dgParamSize)
        amplitude = p[1]/sqrAmplitudeNorm
        means = p[2:nDim+1]
        sigmas = p[nDim+2:dgParamSize]
        print("[");
        amplColor = :normal
        if amplitude^2 <= amplitudeCut
            amplColor = :red
        end
        as = @sprintf("%4.3g", amplitude)
        printstyled(as, color = amplColor)
        print("] [")
        for (i,m) in enumerate(means)
            ms = @sprintf("%4.3g", m)

            print(ms)
            if i!=length(means)
                print(", ")
            else
                print("] [")
            end
        end
        for (i,sg) in enumerate(sigmas)
            sgs = @sprintf("%4.3g", sg)
            sigmaColor = :normal

            if sg <= sigmaCut
                sigmaColor = :red
            end
            printstyled(sgs, color = sigmaColor )
            if i!=length(means)
                print(", ")
            else
                println("]")
            end
        end
    end
end

function printGaussianParamsV3(params::Vector, nDim::Int64 = 1, probHandling::ProbabilityParameterisation = AngleAmplitude, amplitudeCut = 1e-6, sigmaCut = 1e-6)

    dgParamSize =  nDim*2

    if probHandling == SqrtAmplitude
        nGauss =  div(length(params), 2*dgParamSize+2)
        nProbParams = nGauss
        dgAParamSize = nGauss*dgParamSize+nGauss
        amplitudes = params[1:nProbParams]
        probabilities = zero(amplitudes)
        amplitudeGradient = zeros(nProbParams, nGauss)
        sqrtAmplitudeToAngles(amplitudes, probabilities, amplitudeGradient)
    elseif probHandling == AngleAmplitude
        nGauss = div(length(params)+2, 2*dgParamSize+2)
        nProbParams = nGauss -1

        dgAParamSize = nGauss*dgParamSize+nGauss-1
        angles = params[1:nProbParams]
        probabilities = zeros(nGauss)
        amplitudeGradient = zeros(nProbParams, nGauss)
        angleToProbabilityAmplitudeUnpack(angles, probabilities, amplitudeGradient)
#        sqrAmplitudeNorm = sum(x->x^2, params[1:dgParamSize:dgAParamSize] )
    end
    println(sum(probabilities))
    for (a,p) in zip(probabilities, Iterators.partition(params[nProbParams+1:dgAParamSize], dgParamSize))
        amplitude = a
        means = p[1:nDim]
        sigmas = p[nDim+1:dgParamSize]
        print("[");
        amplColor = :normal
        if amplitude <= amplitudeCut
            amplColor = :red
        end
        as = @sprintf("%4.3g", amplitude)
        printstyled(as, color = amplColor)
        print("] [")
        for (i,m) in enumerate(means)
            ms = @sprintf("%4.3g", m)

            print(ms)
            if i!=length(means)
                print(", ")
            else
                print("] [")
            end
        end
        for (i,sg) in enumerate(sigmas)
            sgs = @sprintf("%4.3g", sg)
            sigmaColor = :normal

            if sg <= sigmaCut
                sigmaColor = :red
            end
            printstyled(sgs, color = sigmaColor )
            if i!=length(means)
                print(", ")
            else
                println("]")
            end
        end
    end
end

function printGaussianParamsV4(params::Vector, nDim::Int64 = 1, probHandling::ProbabilityParameterisation = AngleAmplitude, amplitudeCut = 1e-6, sigmaCut = 1e-6)
    dgParamSize =  nDim*2
    dgAParamSize = div(length(params),2)
    avgDeltaMPar = sum(params[dgAParamSize+1:end])/dgAParamSize
    if avgDeltaMPar < 0
        avgDeltaMPar = 0
    end
    function transformDeltaM(par::Float64)
        #return 0.5*tanh(par/7.1) + 0.5
        #return 1/(1+par^2)

        #return exp(par-avgDeltaMPar)
        return exp(par)
    end



    if probHandling == SqrtAmplitude
        nGauss =  div(length(params), 2*dgParamSize+2)
        nProbParams = nGauss
        dgAParamSize = nGauss*dgParamSize+nGauss
        amplitudes = params[1:nProbParams]
        probabilities = zero(amplitudes)
        amplitudeGradient = zeros(nProbParams, nGauss)
#        sqrtAmplitudeToAngles
        amplitudeToProbability(amplitudes, probabilities, amplitudeGradient)
        daparm = params[1:nProbParams]
        datr = @. -log( transformDeltaM(daparm) )
        dprob = abs.(transpose(datr) * amplitudeGradient)

    elseif probHandling == AngleAmplitude
        nGauss = div(length(params)+2, 2*dgParamSize+2)
        nProbParams = nGauss -1

        dgAParamSize = nGauss*dgParamSize+nGauss-1
        angles = params[1:nProbParams]
        probabilities = zeros(nGauss)
        #println(nDim," ",dgParamSize," ",dgAParamSize," ",nGauss," ",nProbParams)
        amplitudeGradient = zeros(nProbParams, nGauss)
        angleToProbabilityAmplitudeUnpack(angles, probabilities, amplitudeGradient)

        if nProbParams!=0
            daparm = params[dgAParamSize+1:dgAParamSize+nProbParams]
            datr = -log.( 0.5*tanh.(0.5.*daparm) .+ 0.5 )
            dprob = abs.(transpose(datr) * amplitudeGradient)
        else
            dprob = 0
        end
        println(dprob)
#        sqrAmplitudeNorm = sum(x->x^2, params[1:dgParamSize:dgAParamSize] )
    end
    println(sum(probabilities))
    for (a,p,da,dp) in zip(probabilities, Iterators.partition(params[nProbParams+1:dgAParamSize], dgParamSize), dprob, Iterators.partition(params[dgAParamSize+nProbParams+1:end],dgParamSize))
        amplitude = a
        means = p[1:nDim]
        sigmas = p[nDim+1:dgParamSize]
        print("[");
        amplColor = :normal
        if amplitude <= amplitudeCut
            amplColor = :red
        end
        as = @sprintf("%4.3g", amplitude)
        printstyled(as, color = amplColor)
        print("] [")
        for (i,m) in enumerate(means)
            ms = @sprintf("%4.3g", m)

            print(ms)
            if i!=length(means)
                print(", ")
            else
                print("] [")
            end
        end
        for (i,sg) in enumerate(sigmas)
            sgs = @sprintf("%4.3g", sg)
            sigmaColor = :normal

            if sg <= sigmaCut
                sigmaColor = :red
            end
            printstyled(sgs, color = sigmaColor )
            if i!=length(means)
                print(", ")
            else
                #println("]")
                print("] {")
            end
        end

        das = @sprintf("%4.3g",da)
        printstyled(das, color = :blue)
        print("} {")
#        dTransf = -log.( 1.0 ./(1.0 .+ dp.^2) )
        dTransf = @. -log(transformDeltaM(dp)) #-log.( 0.5*tanh.(0.5.*dp) .+ 0.5 )
        dmv = dTransf[1:nDim]
        dsv = dTransf[nDim+1:2*nDim]
        for (i,dmean) in enumerate(dmv)
            dmeans = @sprintf("%4.3g",dmean)
            printstyled(dmeans, color = :blue)
            if i!=length(means)
                print(", ")
            else
                #println("]")
                print("} {")
            end
        end

        for (i,dsigma) in enumerate(dsv)
            dsigmas = @sprintf("%4.3g",dsigma)
            printstyled(dsigmas, color = :blue)
            if i!=length(means)
                print(", ")
            else
                #println("]")
                println("}")
            end
        end
    end
end


function plotGaussian( params::Vector, nDim = 2, lines = true, nGauss = -1)

    dgParamSize = nDim*2+1
    if nGauss == -1
        nGauss = div(length(params) , (2*dgParamSize))
    end
    dgAParamSize = nGauss*dgParamSize
    dgA = Array{diagonalGauss{Float64}, 1}()
    sqrAmplitudeNorm = sum(x->x^2, params[1:dgParamSize:dgAParamSize] )
    for p in Iterators.partition(params[1:dgAParamSize], dgParamSize)
        p2 = copy(p)
        p2[1]/=sqrAmplitudeNorm
        println(p2)
        push!(dgA, diagonalGauss(collect(p)))
    end
    contour!(-5:0.1:10,-5:0.1:10, (x,y)->log(probability(dgA, [x,y])), fill = !lines)
end

function plotGaussianV3( params::Vector, nDim = 2, lines = true, pl = plot(), plotRanges=(-5:0.1:10,-5:0.1:10), nGauss = -1)

    dgParamSize = nDim*2
    if nGauss == -1
        nGauss = div(length(params)+2 , (2*dgParamSize+2))
    end
    dgAParamSize = nGauss*dgParamSize+nGauss-1
    dgA = dgAConstructSwitch(params[1:dgAParamSize], nDim, AngleAmplitude)
    #dgA = Array{diagonalGauss{Float64}, 1}()
    # sqrAmplitudeNorm = sum(x->x^2, params[1:dgParamSize:dgAParamSize] )
    # for p in Iterators.partition(params[1:dgAParamSize], dgParamSize)
    #     p2 = copy(p)
    #     p2[1]/=sqrAmplitudeNorm
    #     println(p2)
    #     push!(dgA, diagonalGauss(collect(p)))
    # end
    contour!(pl,plotRanges[1],plotRanges[2] , (x,y)->log(probability(dgA, [x,y])), fill = !lines, label = "log ΣGauss")
    return pl
end

function entropyV3(params::Vector{Float64}, data::Array{Float64,2})
    nDim  = size(data)[1]
    dgParamSize = nDim*2
    nGauss = div(length(params)+2 , (2*dgParamSize+2))
    dgAParamSize = nGauss*dgParamSize+nGauss-1
    dgA = dgAConstructSwitch(params[1:dgAParamSize], nDim, AngleAmplitude)
    ll = sum(x->log(probability(dgA,copy(x))), eachcol(data) )
    return -ll
end

function histoGaussian(params::Vector)
    dgParamSize = 3
    nGauss = div(length(params), 2*dgParamSize)
    dgAParamSize = nGauss*dgParamSize
    dgA = dgAConstruct(params[1:dgAParamSize],1)
    sqrAplitudeNorm = sum(x->x^2, params[1:dgParamSize:dgAParamSize])
    printGaussianParams(params,1)
    for p in Iterators.partition(params[1:dgAParamSize], dgParamSize)
        p2 = copy(p)
        p2[1]/=sqrAplitudeNorm
#        println(p2)
    end
    plot!(-5:0.01:10, x->probability(dgA, [x]))
end

function plotGausses(means, sigmas, amplitudes, colorpalette, p = plot(), plotRange = -10:0.1:10, sumOnly = false, sumLineWidth = 1)
    amplNorm = sum(amplitudes)
    #p = plot()
    if !sumOnly
        for (i,(m,s,a)) in enumerate(zip(means, sigmas, amplitudes))
            #push!(p ,plot(p,x->a*gauss([m,s],x)/amplNorm, plotRange, linewidth = 3))
            gaussLabel = @sprintf("Gauss %i",i)
            plot!(x->a*gauss([m,s],x)/amplNorm, plotRange, linewidth = 3, color = colorpalette[i],linestyle = :dot, label=gaussLabel)
        end
    end
    plot!(p, x->sum(map( (m,s,a)->a*gauss([m,s],x)/amplNorm, means, sigmas, amplitudes )), plotRange, linewidth = sumLineWidth, label="∑Gauss")
    return p
end

function plotGausses(p::Array, pl = plot(), plotRange = -10:0.1:10, sumOnly = false, sumLineWidth = 1)
    #mod(length(p) , 6) == 0 || error("length(p) must be divisible by 6")
    dgParamSize = 3
    nGauss = div(length(p), 2*dgParamSize)
    dgAParamSize = nGauss*dgParamSize
    means = zeros(length(p)÷6)
    sigmas = zeros(length(p)÷6)
    amplitudes = zeros(length(p)÷6)
    sqrAmplitudeNorm = sum(x->x^2, p[1:3:dgAParamSize] )
    printGaussianParams(p,1)
    if nGauss > 1
        colorpalette = [RGB(0.0, i, 0.0) for i in range(1.0, 0.0, length=nGauss)]
    else
        colorpalette = [RGB(0.0, 1.0, 0.0)]
    end
    for (i, (a,m, s)) in enumerate( Iterators.partition(p[1:dgAParamSize], dgParamSize) )
        means[i] = m
        sigmas[i] = s
        amplitudes[i] = a*a/sqrAmplitudeNorm

    end
    plotGausses(means, sigmas, amplitudes, colorpalette, pl, plotRange, sumOnly, sumLineWidth )
end

function plotGaussesV2(p::Array, pl = plot(), plotRange = -10:0.1:10, sumOnly = false, sumLineWidth = 1)
    #mod(length(p) , 6) == 0 || error("length(p) must be divisible by 6")
    dgParamSize = 3
    nGauss = div(length(p), 2*dgParamSize)
    dgAParamSize = nGauss*dgParamSize
    means = p[nGauss+1:2:dgAParamSize]
    sigmas = p[nGauss+2:2:dgAParamSize]
    amplitudes = p[1:nGauss].^2
    sqrAmplitudeNorm = sum( amplitudes )
    #printGaussianParams(p,1)
    if nGauss > 1
        colorpalette = [RGB(0.0, i, 0.0) for i in range(1.0, 0.0, length=nGauss)]
    else
        colorpalette = [RGB(0.0, 1.0, 0.0)]
    end
    # for (i, (a,m, s)) in enumerate( Iterators.partition(p[1:dgAParamSize], dgParamSize) )
    #     means[i] = m
    #     sigmas[i] = s
    #     amplitudes[i] = a*a/sqrAmplitudeNorm
    #
    # end
    plotGausses(means, sigmas, amplitudes, colorpalette, pl, plotRange, sumOnly, sumLineWidth)
end

function plotGaussesV3(p::Array, pl = plot(), plotRange = -10:0.1:10, sumOnly = false, sumLineWidth = 1)
    #mod(length(p) , 6) == 0 || error("length(p) must be divisible by 6")
    dgParamSize = 3
    nGauss = div(length(p)+2, 2*dgParamSize)
    dgAParamSize = nGauss*dgParamSize-1
    means = p[nGauss:2:dgAParamSize]
    sigmas = p[nGauss+1:2:dgAParamSize]
    angles = p[1:nGauss-1]
    amplitudes = zeros(nGauss)
    amplitudeGradient=zeros(nGauss-1, nGauss)
    angleToProbabilityAmplitudeUnpack(angles,amplitudes, amplitudeGradient)
#    amplitudes = p[1:nGauss].^2
#    sqrAmplitudeNorm = sum( amplitudes )
#    printGaussianParams(p,1)
    if nGauss > 1
        colorpalette = [RGB(0.0, i, 0.0) for i in range(1.0, 0.0, length=nGauss)]
    else
        colorpalette = [RGB(0.0, 1.0, 0.0)]
    end
    # for (i, (a,m, s)) in enumerate( Iterators.partition(p[1:dgAParamSize], dgParamSize) )
    #     means[i] = m
    #     sigmas[i] = s
    #     amplitudes[i] = a*a/sqrAmplitudeNorm
    #
    # end
    plotGausses(means, sigmas, amplitudes, colorpalette, pl, plotRange, sumOnly, sumLineWidth)
end

function dgSumLogCDF(dgA::Array{diagonalGauss{Float64}, 1},
    dgABounds::Vector{Float64}, data::Array{Float64,2},
    logarithmicdeDeltaRange = true,
    linearLogApprox = false,
    varyDx = true )

    nGauss = length(dgA)
    nDim = length(dgA[1].means)
    dFda = zeros(nGauss, nDim)
    dFdm = zeros(nGauss, nDim, nDim)
    dFds = zeros(nGauss, nDim, nDim)
    dFdx = zeros(nDim, nDim)
    dFdx_ortho = zeros(nDim, nDim)


    Da = zeros(nGauss)
    Dm = zeros(nGauss, nDim)
    Ds = zeros(nGauss, nDim)
    chunkSize = 2*nDim+1

    sumlogp = 0.0
    largestScale    = -1.0
    dgAParamSize = nGauss*(2*nDim + 1)

    for (k,p) in enumerate(Iterators.partition(dgABounds[1:dgAParamSize], chunkSize))

        if !logarithmicdeDeltaRange
            Da[k] = p[1]^2
            Dm[k,:] = p[2:nDim+1].^2
            Ds[k,:] = p[nDim+2:end].^2
        else
            Da[k] = exp(p[1])
            for i in 1:div(chunkSize,2)
                Dm[k,i] = exp(p[i+1])
                Ds[k,i] = exp(p[nDim+1+i])
            end
        end

    end

    Dx = zeros(nDim)
    if varyDx
        if logarithmicdeDeltaRange
            Dx[:] = dgABounds[dgAParamSize+1:dgAParamSize+nDim]
            logDXnorm = sum(Dx)/nDim
            Dx = exp.(Dx .- logDXnorm)
        end
    end

    for d in eachcol(data)
        dgMarginalCDFDerivative(dgA, copy(d), dFda, dFdm, dFds)
        for nu in 1:nDim, lambda in 1:nDim
            dFdx[nu,lambda] = 0
        end
        # if varyDx
        #     for nu in 1:nDim, k in 1:nGauss, lambda in 1:nDim
        #         dFdx[nu, lambda] -= dFdm[k, nu, lambda] * Dx[lambda]
        #     end
        # else
        #     for nu in 1:nDim, k in 1:nGauss, lambda in 1:nDim
        #         dFdx[nu, lambda] -= dFdm[k, nu, lambda]
        #     end
        # end

        for nu in 1:nDim, k in 1:nGauss, lambda in 1:nDim
            dFdx[nu, lambda] -= dFdm[k, nu, lambda]
        end

        if varyDx
            for nu in 1:nDim, lambda in 1:nDim
                 dFdx[nu, lambda] *= Dx[nu]
            end
        end

        perturbation = 0.0
        dFdx_ortho = inv(dFdx)

        for k in 1:nGauss
            for nu in 1:nDim
                dotProd = 0.0
                for lambda in 1:nDim
                    dotProd += dFdx_ortho[lambda, nu] * dFda[k,lambda]
                end
                perturbation -= abs(dotProd)*Da[k]
            end

            for nu in 1:nDim
                for lambda in 1:nDim
                    dotProdM = 0.0
                    dotProdS = 0.0
                    for kappa in 1:nDim
                        dotProdM += dFdx_ortho[kappa ,nu] * dFdm[k, kappa, lambda]
                        dotProdS += dFdx_ortho[kappa, nu] * dFds[k, kappa, lambda]
                    end
                    perturbation -= abs(dotProdM)*Dm[k, lambda]
                    perturbation -= abs(dotProdS)*Ds[k, lambda]
                end
            end
        end


    # if false
    # getOrthogonalSubspaces(dFdx, dFdx_ortho, nDim)
    # detLikelihood = det(dFdx)
    # if varyDx
    #     for nu in 1:nDim, lambda in 1:nDim
    #          dFdx[nu, lambda] *= Dx[nu]
    #     end
    # end
    #     for k in 1:nGauss
    #         asign = 0.0
    #         for nu in 1:nDim
    #             dFda[k, nu] *=  Da[k]
    #             for lambda in 1:nDim
    #                 dFdm[k, nu , lambda ] *= Dm[k, lambda] #free nu
    #                 dFds[k, nu , lambda ] *= Ds[k, lambda] #free nu
    #                 asign += dFdx_ortho[nu, lambda] * dFda[k, nu]
    #
    #             end
    #         end
    #         asign = sign(asign)
    #         #                asign = sign(dot(dFdx_ortho[:, kappa], dFda[k,:]))
    #         for nu in 1:nDim
    #             for kappa in 1:nDim
    #                 dFdx[nu,kappa] -= asign * dFda[k,nu]
    #             end
    #         end
    #
    #
    #         for kappa in 1:nDim, lambda in 1:nDim
    #             msign = 0.0
    #             ssign = 0.0
    #             for nu in 1:nDim
    #                 msign += dFdx_ortho[nu, kappa]* dFdm[k,nu,lambda]
    #                 ssign += dFdx_ortho[nu, kappa]* dFds[k,nu,lambda]
    #             end
    #             msign = sign(msign)
    #             ssign = sign(ssign)
    #             for nu in 1:nDim
    #                 dFdx[nu,kappa] -= msign *dFdm[k, nu , lambda ]
    #                 dFdx[nu,kappa] -= ssign *dFds[k, nu , lambda ]
    #             end
    #         end
    #     end
    # end

        # detDu = det(dFdx)
        # scale = abs(detDu - detLikelihood)/abs(detLikelihood)

    perturbation += 1.0
    #println(perturbation)
    scale = abs(perturbation)

    detLikelihood = det(dFdx)
    detDu = detLikelihood*perturbation
        if largestScale < scale
            largestScale = scale
        end
        if !linearLogApprox
            #if sign(detDu) != sign(detLikelihood)
            if perturbation <= 0.0
                sumlogp = Inf
            else
                sumlogp -= log(abs(detDu))
                #log(abs(detLikelihood))#
                #
            end
        else
            volume = sign(detLikelihood)*detDu
            if volume < 0.
                sumlogp -= 1000.0*volume
            else
                sumlogp -=volume
            end
        end
        global dScale = largestScale
    end
    if isnan(sumlogp)
        #println(dgA)
    end

    return sumlogp -sum(log.(Da)) -sum(log.(Dm)) -sum(log.(Ds))

end

function dgSumLogCDFLocalDxVary(dgA::Array{diagonalGauss{Float64}, 1},
    dgABounds::Vector{Float64}, data::Array{Float64,2},
    logarithmicdeDeltaRange = true,
    linearLogApprox = false,
    varyDx = true )

    nGauss = length(dgA)
    nDim = length(dgA[1].means)
    dFda = zeros(nGauss, nDim)
    dFdm = zeros(nGauss, nDim, nDim)
    dFds = zeros(nGauss, nDim, nDim)
    dFdx = zeros(nDim, nDim)
    dFdx_inv = zeros(nDim, nDim)


    Da = zeros(nGauss)
    Dm = zeros(nGauss, nDim)
    Ds = zeros(nGauss, nDim)
    chunkSize = 2*nDim+1

    sumlogp = 0.0
    largestScale    = -1.0
    dgAParamSize = nGauss*(2*nDim + 1)

    perturbationVec = zeros(nDim)

    for (k,p) in enumerate(Iterators.partition(dgABounds[1:dgAParamSize], chunkSize))

        if !logarithmicdeDeltaRange
            Da[k] = p[1]^2
            Dm[k,:] = p[2:nDim+1].^2
            Ds[k,:] = p[nDim+2:end].^2
        else
            Da[k] = exp(p[1])
            for i in 1:div(chunkSize,2)
                Dm[k,i] = exp(p[i+1])
                Ds[k,i] = exp(p[nDim+1+i])
            end
        end

    end

    # Dx = zeros(nDim)
    # if varyDx
    #     if logarithmicdeDeltaRange
    #         Dx[:] = dgABounds[dgAParamSize+1:dgAParamSize+nDim]
    #         logDXnorm = sum(Dx)/nDim
    #         Dx = exp.(Dx .- logDXnorm)
    #     end
    # end

    for d in eachcol(data)
        dgMarginalCDFDerivative(dgA, copy(d), dFda, dFdm, dFds)
        for nu in 1:nDim, lambda in 1:nDim
            dFdx[nu,lambda] = 0
        end
        # if varyDx
        #     for nu in 1:nDim, k in 1:nGauss, lambda in 1:nDim
        #         dFdx[nu, lambda] -= dFdm[k, nu, lambda] * Dx[lambda]
        #     end
        # else
        #     for nu in 1:nDim, k in 1:nGauss, lambda in 1:nDim
        #         dFdx[nu, lambda] -= dFdm[k, nu, lambda]
        #     end
        # end

        for nu in 1:nDim, k in 1:nGauss, lambda in 1:nDim
            dFdx[nu, lambda] -= dFdm[k, nu, lambda]
        end

        # if varyDx
        #     for nu in 1:nDim, lambda in 1:nDim
        #          dFdx[nu, lambda] *= Dx[nu]
        #     end
        # end

        for nu in 1:nDim
            perturbationVec[nu] = 0.0
        end

        dFdx_inv = inv(dFdx)

        for k in 1:nGauss
            for nu in 1:nDim
                dotProd = 0.0
                for lambda in 1:nDim
                    dotProd += dFdx_inv[lambda, nu] * dFda[k,lambda]
                end
                perturbationVec[nu] += abs(dotProd)*Da[k]
            end

            for nu in 1:nDim
                for lambda in 1:nDim
                    dotProdM = 0.0
                    dotProdS = 0.0
                    for kappa in 1:nDim
                        dotProdM += dFdx_inv[kappa ,nu] * dFdm[k, kappa, lambda]
                        dotProdS += dFdx_inv[kappa, nu] * dFds[k, kappa, lambda]
                    end
                    perturbationVec[nu] += abs(dotProdM)*Dm[k, lambda]
                    perturbationVec[nu] += abs(dotProdS)*Ds[k, lambda]
                end
            end
        end
        perturbationNorm = prod(perturbationVec)^(1.0/nDim)

      #print(perturbationNorm, " ")

    # if false
    # getOrthogonalSubspaces(dFdx, dFdx_ortho, nDim)
    # detLikelihood = det(dFdx)
    # if varyDx
    #     for nu in 1:nDim, lambda in 1:nDim
    #          dFdx[nu, lambda] *= Dx[nu]
    #     end
    # end
    #     for k in 1:nGauss
    #         asign = 0.0
    #         for nu in 1:nDim
    #             dFda[k, nu] *=  Da[k]
    #             for lambda in 1:nDim
    #                 dFdm[k, nu , lambda ] *= Dm[k, lambda] #free nu
    #                 dFds[k, nu , lambda ] *= Ds[k, lambda] #free nu
    #                 asign += dFdx_ortho[nu, lambda] * dFda[k, nu]
    #
    #             end
    #         end
    #         asign = sign(asign)
    #         #                asign = sign(dot(dFdx_ortho[:, kappa], dFda[k,:]))
    #         for nu in 1:nDim
    #             for kappa in 1:nDim
    #                 dFdx[nu,kappa] -= asign * dFda[k,nu]
    #             end
    #         end
    #
    #
    #         for kappa in 1:nDim, lambda in 1:nDim
    #             msign = 0.0
    #             ssign = 0.0
    #             for nu in 1:nDim
    #                 msign += dFdx_ortho[nu, kappa]* dFdm[k,nu,lambda]
    #                 ssign += dFdx_ortho[nu, kappa]* dFds[k,nu,lambda]
    #             end
    #             msign = sign(msign)
    #             ssign = sign(ssign)
    #             for nu in 1:nDim
    #                 dFdx[nu,kappa] -= msign *dFdm[k, nu , lambda ]
    #                 dFdx[nu,kappa] -= ssign *dFds[k, nu , lambda ]
    #             end
    #         end
    #     end
    # end

        # detDu = det(dFdx)
        # scale = abs(detDu - detLikelihood)/abs(detLikelihood)
    #println(sum(perturbationVec))

    perturbation = 1.0 - nDim * perturbationNorm

    #println(perturbation)
    scale = abs(perturbation)

    detLikelihood = det(dFdx)
    detDu = detLikelihood*perturbation
        if largestScale < scale
            largestScale = scale
        end
        if !linearLogApprox
            #if sign(detDu) != sign(detLikelihood)
            if perturbation <= 0.0
                sumlogp = Inf
            else
                sumlogp -= log(abs(detDu))
                #log(abs(detLikelihood))#
                #
            end
        else
            volume = sign(detLikelihood)*detDu
            if volume < 0.
                sumlogp -= 1000.0*volume
            else
                sumlogp -=volume
            end
        end
        global dScale = largestScale
    end
    if isnan(sumlogp)
        #println(dgA)
    end

    return sumlogp -sum(log.(Da)) -sum(log.(Dm)) -sum(log.(Ds))
end

@polly @inbounds function dgQ(dgAv::Vector{Float64},
             dgABounds::Vector{Float64},
             probHandling::ProbabilityParameterisation,
             nDim::Int64, nGauss::Int64, nProbParams::Int64,
             mA::marginalCDFDerivativeAllocations{Float64},
             #dgParamSize::Int64,
             dgAParamSize::Int64,
             data::Array{Float64,2}, varyDX = false)

    avgDMParams = sum(dgABounds)/dgAParamSize
    if avgDMParams < 0
        avgDMParams = 0
    end
    function transformDeltaM(par::Float64)
        # if par<0.0
        #     return 0.5*exp(par)
        # else
        #     #return 1.0-0.5exp(-par)
        #     return tanh(par/7.1) # bounds are set to +137, and 137/7.1 is nearly the maximal value not giving 1
        # end
        #return 0.5*erf(par/10.0)+0.5
        #return 1/(1+par^2)
        #return 0.5*tanh(par/7.1) + 0.5

        #return exp(par-avgDMParams)
        return exp(par)

    end
    #end
    #mA = marginalCDFDerivativeAllocations{Float32}(nDim,nGauss,nProbParams)
    #dgAv = Vector{Float32}(dgAvIn)
    if nProbParams > 0
        mA.probParams .= dgAv[1:nProbParams]
    end
    mA.normalizedAmplitudes .= 0
    mA.amplitudeGradient .= 0
    mA.perturbationSumLog = 0

    if nProbParams != 0
        #mA.Da .= exp.(-dgABounds[1:nProbParams].^2)
        #mA.Da .= 1 ./ (1 .+ dgABounds[1:nProbParams].^2)
        @. mA.Da .= transformDeltaM(dgABounds[1:nProbParams])
    else
        mA.Da .= 0
    end

    for (k,p) in enumerate(Iterators.partition(dgABounds[nProbParams+1:dgAParamSize], 2*nDim))
        #mA.Dm[k,:] = exp.(-p[1:nDim].^2)
        #mA.Dm[k,:] .= 1 ./ (1 .+ p[1:nDim].^2)
        @. mA.Dm[k,:] = transformDeltaM(p[1:nDim])
        #mA.Ds[k,:] .= exp.(-p[nDim+1:end].^2)
        #mA.Ds[k,:] .= 1 ./ (1 .+ p[nDim+1:end].^2)
        @. mA.Ds[k,:] = transformDeltaM(p[nDim+1:end])

    end
    dgA = dgAConstructSwitch(dgAv, nDim, probHandling, mA.normalizedAmplitudes, mA.amplitudeGradient)

    sumlogp = 0.0
    largestScale    = -1.0
    #    dgAParamSize = nGauss*(2*nDim + 1)
    for (index,d) in enumerate(eachcol(data))
        dgExternalAllocationsAmplitudesMarginalCDFDerivative(dgA, copy(d), mA)
        for nu in 1:nDim, k in 1:nGauss, lambda in 1:nDim
            mA.dFdx[nu, lambda] -= mA.dFdm[k, nu, lambda]
        end
        #println(mA.dFdx)
        mA.dFdx_inv .= inv(mA.dFdx)
        for k in 1:nProbParams
            for nu in 1:nDim
                dotProd = 0.0
                for lambda in 1:nDim
                    dotProd += mA.dFdx_inv[lambda, nu] * mA.dFda[k,lambda]
                end
                mA.perturbationVec[nu] += abs(dotProd)*mA.Da[k]#*1e-6 #bookmark
            end
        end

        for k in 1:nGauss
            for nu in 1:nDim
                for lambda in 1:nDim
                    dotProdM = 0.0
                    dotProdS = 0.0
                    for kappa in 1:nDim
                        dotProdM += mA.dFdx_inv[kappa ,nu] * mA.dFdm[k, kappa, lambda]
                        dotProdS += mA.dFdx_inv[kappa, nu] * mA.dFds[k, kappa, lambda]
                    end
                    mA.perturbationVec[nu] += abs(dotProdM)*mA.Dm[k, lambda]
                    mA.perturbationVec[nu] += abs(dotProdS)*mA.Ds[k, lambda]
                end
            end
        end

        if varyDX
            for nu in 1:nDim
                mA.perturbationSumLog += log(mA.perturbationVec[nu])/nDim
            end
        else
            pert = 1.0 - sum(mA.perturbationVec)
            if pert <= 0
                sumlogp = Inf
                #println("Inf @ index ",index)
            else
                sumlogp -= log(pert)
            end
        end

        maxLLindex = 1
        maxLL = mA.logProbabilities[1]
        for k in 2:nGauss
            if maxLL < mA.logProbabilities[k]
                maxLLindex = k
                maxLL = mA.logProbabilities[k]
            end
        end

        if maxLL < -500.0
            sumlogp -= maxLL
            for k in filter(!isequal(maxLLindex), 1:nGauss)
                sumlogp-=exp(mA.logProbabilities[k]-maxLL)
            end
        else
            logProb = 0.0
            for k in 1:nGauss
                #sumlogp-=logProbabilities[k]
                logProb+=exp(mA.logProbabilities[k])
            end
            sumlogp -= log(logProb)
            # if logProb != log(abs(det(dFdx)))
            #     println("logProb != log(abs(det(dFdx)))")
            #     println(logProb, " !=  ",  log(abs(det(dFdx))))
            #     error("!")
            # end
        end
        #println(sumlogp)
    end

    if false && isnan(sumlogp)
        println("***** sl == NaN *****")
        println("dgA: " , dgA)
        println("dgABounds: ", dgABounds)
    end
    nData =  size(data)[2]

    global dScale = mA.perturbationSumLog/nData + log(nDim)
    perturbationFactor = 1.0 - exp(mA.perturbationSumLog/nData)*nDim
    perturbationTotal = -Inf
    if perturbationFactor > 0
        perturbationTotal = nData*log(perturbationFactor)
    end
    if varyDX
        sumlogp -= perturbationTotal
    end

    for e in mA.Da[1:nProbParams]
        sumlogp-=log(e)
    end

    for e in mA.Dm
        sumlogp-=log(e)
    end

    for e in mA.Ds
        sumlogp-=log(e)
    end

    if isinf(sumlogp)
        sumlogp = sign(sumlogp)*9.9e99
    end
    return sumlogp
end


function dgQold(dgAv::Vector{Float64},
             dgABounds::Vector{Float64},
             probHandling::ProbabilityParameterisation,
             nDim::Int64, nGauss::Int64, nProbParams::Int64,
             #dgParamSize::Int64,
             dgAParamSize::Int64,
             data::Array{Float64,2})


    #dFda = zeros(nProbParams, nDim)
    dFdm = zeros(nGauss, nDim, nDim)
    dFds = zeros(nGauss, nDim, nDim)
    dFdx = zeros(nDim, nDim)
    dFdx_inv = zeros(nDim, nDim)
    logProbabilities = zeros(nGauss)

    Dm = zeros(nGauss, nDim)
    Ds = zeros(nGauss, nDim)


    probParams = dgAv[1:nProbParams]
    amplitudes = zeros(nGauss)
    if nProbParams == 0
        amplitudeGradient = zeros(1, nGauss)
        dFda = zeros(1, nDim)
        Da = zeros(1)
    else
        amplitudeGradient = zeros(nProbParams, nGauss)
        dFda = zeros(nProbParams, nDim)
        Da = exp.(dgABounds[1:nProbParams])
    end

    for (k,p) in enumerate(Iterators.partition(dgABounds[nProbParams+1:dgAParamSize], 2*nDim))

        Dm[k,:] = exp.(p[1:nDim])
        Ds[k,:] = exp.(p[nDim+1:end])
    end
    dgA = dgAConstructSwitch(dgAv, nDim, probHandling, amplitudes, amplitudeGradient)

    sumlogp = 0.0
    largestScale    = -1.0
#    dgAParamSize = nGauss*(2*nDim + 1)

    perturbationVec = zeros(nDim)
    perturbationSumLog = 0.0

    for d in eachcol(data)
        dgExternalAmplitudesMarginalCDFDerivative(dgA, amplitudes, amplitudeGradient, copy(d), dFda, dFdm, dFds, logProbabilities)
        for nu in 1:nDim, lambda in 1:nDim
            dFdx[nu,lambda] = 0
        end

        for nu in 1:nDim, k in 1:nGauss, lambda in 1:nDim
            dFdx[nu, lambda] -= dFdm[k, nu, lambda]
        end

        for nu in 1:nDim
            perturbationVec[nu] = 0.0
        end

        dFdx_inv = inv(dFdx)

        for k in 1:nProbParams
            for nu in 1:nDim
                dotProd = 0.0
                for lambda in 1:nDim
                    dotProd += dFdx_inv[lambda, nu] * dFda[k,lambda]
                end
                perturbationVec[nu] += abs(dotProd)*Da[k]
            end
        end

        for k in 1:nGauss
            for nu in 1:nDim
                for lambda in 1:nDim
                    dotProdM = 0.0
                    dotProdS = 0.0
                    for kappa in 1:nDim
                        dotProdM += dFdx_inv[kappa ,nu] * dFdm[k, kappa, lambda]
                        dotProdS += dFdx_inv[kappa, nu] * dFds[k, kappa, lambda]
                    end
                    perturbationVec[nu] += abs(dotProdM)*Dm[k, lambda]
                    perturbationVec[nu] += abs(dotProdS)*Ds[k, lambda]
                end
            end
        end

        for nu in 1:nDim
            perturbationSumLog += log(perturbationVec[nu])/nDim
        end


        maxLLindex = 1
        maxLL = logProbabilities[1]
        for k in 2:nGauss
            if maxLL < logProbabilities[k]
                maxLLindex = k
                maxLL = logProbabilities[k]
            end
        end

        if maxLL < -500.0
            sumlogp -= maxLL
            for k in filter(!isequal(maxLLindex), 1:nGauss)
                sumlogp-=exp(logProbabilities[k]-maxLL)
            end
        else
            logProb = 0.0
            for k in 1:nGauss
                #sumlogp-=logProbabilities[k]
                logProb+=exp(logProbabilities[k])
            end
            sumlogp -= log(logProb)
            # if logProb != log(abs(det(dFdx)))
            #     println("logProb != log(abs(det(dFdx)))")
            #     println(logProb, " !=  ",  log(abs(det(dFdx))))
            #     error("!")
            # end
        end
    end

    if false && isnan(sumlogp)
        println("***** sl == NaN *****")
        println("dgA: " , dgA)
        println("dgABounds: ", dgABounds)
    end

    nData =  size(data)[2]

    global dScale = perturbationSumLog/nData + log(nDim)
    perturbationFactor = 1.0 - exp(perturbationSumLog/nData)*nDim
    perturbationTotal = -Inf
    if perturbationFactor > 0
        perturbationTotal = nData*log(perturbationFactor)
    end

    #println(perturbationFactor)
    for e in Da[1:nProbParams]
        sumlogp-=log(e)
    end

    for e in Dm
        sumlogp-=log(e)
    end

    for e in Ds
        sumlogp-=log(e)
    end
    sumlogp -= perturbationTotal
    if isinf(sumlogp)
        sumlogp = sign(sumlogp)*9.9e99
    end
    return sumlogp
end

function dgSumLogCDFTotalDxVary(dgA::Array{diagonalGauss{Float64}, 1},
    dgABounds::Vector{Float64}, data::Array{Float64,2},
    logarithmicdeDeltaRange = true,
    linearLogApprox = false,
    varyDx = true )

    nGauss = length(dgA)
    nDim = length(dgA[1].means)
    dFda = zeros(nGauss, nDim)
    dFdm = zeros(nGauss, nDim, nDim)
    dFds = zeros(nGauss, nDim, nDim)
    dFdx = zeros(nDim, nDim)
    dFdx_inv = zeros(nDim, nDim)
    logProbabilities = zeros(nGauss)

    Da = zeros(nGauss)
    Dm = zeros(nGauss, nDim)
    Ds = zeros(nGauss, nDim)
    chunkSize = 2*nDim+1

    sumlogp = 0.0
    largestScale    = -1.0
    dgAParamSize = nGauss*(2*nDim + 1)

    perturbationVec = zeros(nDim)
    perturbationSumLog = 0.0

    for (k,p) in enumerate(Iterators.partition(dgABounds[1:dgAParamSize], chunkSize))

        if !logarithmicdeDeltaRange
            Da[k] = p[1]^2
            Dm[k,:] = p[2:nDim+1].^2
            Ds[k,:] = p[nDim+2:end].^2
        else
            Da[k] = exp(p[1])
            for i in 1:div(chunkSize,2)
                Dm[k,i] = exp(p[i+1])
                Ds[k,i] = exp(p[nDim+1+i])
            end
        end

    end

    for d in eachcol(data)
        dgMarginalCDFDerivative(dgA, copy(d), dFda, dFdm, dFds, logProbabilities)
        for nu in 1:nDim, lambda in 1:nDim
            dFdx[nu,lambda] = 0
        end

        for nu in 1:nDim, k in 1:nGauss, lambda in 1:nDim
            dFdx[nu, lambda] -= dFdm[k, nu, lambda]
        end

        for nu in 1:nDim
            perturbationVec[nu] = 0.0
        end

        dFdx_inv = inv(dFdx)

        for k in 1:nGauss
            for nu in 1:nDim
                dotProd = 0.0
                for lambda in 1:nDim
                    dotProd += dFdx_inv[lambda, nu] * dFda[k,lambda]
                end
                perturbationVec[nu] += abs(dotProd)*Da[k]
            end

            for nu in 1:nDim
                for lambda in 1:nDim
                    dotProdM = 0.0
                    dotProdS = 0.0
                    for kappa in 1:nDim
                        dotProdM += dFdx_inv[kappa ,nu] * dFdm[k, kappa, lambda]
                        dotProdS += dFdx_inv[kappa, nu] * dFds[k, kappa, lambda]
                    end
                    perturbationVec[nu] += abs(dotProdM)*Dm[k, lambda]
                    perturbationVec[nu] += abs(dotProdS)*Ds[k, lambda]
                end
            end
        end
        #perturbationNorm = prod(perturbationVec)^(1.0/nDim)

        for nu in 1:nDim
            perturbationSumLog += log(perturbationVec[nu])/nDim
        end

        maxLLindex = 1
        maxLL = logProbabilities[1]
        for k in 2:nGauss
            if maxLL < logProbabilities[k]
                maxLLindex = k
                maxLL = logProbabilities[k]
            end
        end

        if maxLL < -500.0
            sumlogp -= maxLL
            for k in filter(!isequal(maxLLindex), 1:nGauss)
                sumlogp-=exp(logProbabilities[k]-maxLL)
            end
        else
            logProb = 0.0
            for k in 1:nGauss
                #sumlogp-=logProbabilities[k]
                logProb+=exp(logProbabilities[k])
            end
            sumlogp -= log(logProb)
            # if logProb != log(abs(det(dFdx)))
            #     println("logProb != log(abs(det(dFdx)))")
            #     println(logProb, " !=  ",  log(abs(det(dFdx))))
            #     error("!")
            # end
        end


        #sumlogp -= log(abs(det(dFdx)))
    end
    if false && isnan(sumlogp)
        println("***** sl == NaN *****")
        println("dgA: " , dgA)
        println("dgABounds: ", dgABounds)
    end

    nData =  size(data)[2]
#    dxVolume = 1.0
    global dScale = perturbationSumLog/nData + log(nDim)
    perturbationFactor = 1.0 - exp(perturbationSumLog/nData)*nDim
    perturbationTotal = -Inf
    if perturbationFactor > 0
        perturbationTotal = nData*log(perturbationFactor)
    end

    #println(perturbationFactor)
    for e in Da
        sumlogp-=log(e)
    end

    for e in Dm
        sumlogp-=log(e)
    end

    for e in Ds
        sumlogp-=log(e)
    end
    sumlogp -= perturbationTotal
    if isinf(sumlogp)
        sumlogp = sign(sumlogp)*9.9e99
    end
    return sumlogp
    #return sumlogp -sum(log.(Da)) -sum(log.(Dm)) -sum(log.(Ds)) - perturbationTotal
end


function gauss(p::Vector, x)
    length(p) == 2 || error("length(p) must be 2")
    res = exp(-(x-p[1])^2/2.0/p[2]^2)/sqrt(2.0*pi*p[2]^2)
    return res
end

function testBoundaries(startVector::Vector{Float64}, lowerBounds::Vector{Float64},upperBounds::Vector{Float64})
    sum([ (s<=u) && (s>=l) for (s,l,u) in zip(startVector, lowerBounds, upperBounds)])
end

function QOptimizer(data::Array{Float64,2},
    nGauss::Int,
    callLimits::Vector{Int} = [1000,1000],
    probHandling::ProbabilityParameterisation=SqrtAmplitude,
    startParameters::Union{Nothing,Vector{Float64}}=nothing,
    fullOpt = true)

    nDim = size(data)[1]
    nData = size(data)[2]
    dgParamSize = 2*nDim

    if startParameters!=nothing
        if probHandling == SqrtAmplitude
            nGauss = div(length(startParameters),2*dgParamSize+2  )
        elseif probHandling == AngleAmplitude
            nGauss = div(length(startParameters)+2,2*dgParamSize+2  )
        end
        println("nGauss = " , nGauss, " inferred from startParameters" )
    else
        fullOpt = true
    end

    if probHandling == SqrtAmplitude
        nProbParams = nGauss

    else
        if probHandling == AngleAmplitude
            nProbParams = nGauss -1
        end
    end

    dgAParamSize = dgParamSize*nGauss + nProbParams

    fTolerance = 1e-15
    callCounter = 0

    mA = marginalCDFDerivativeAllocations{Float64}(nDim, nGauss, nProbParams)

    function OptimizeMe(algo::Symbol,
                startVector::Vector{Float64},
                lowerBounds::Vector{Float64},
                upperBounds::Vector{Float64},
                fTolerance::Float64, callLimit::Int64, fullOptimization = true, verbose = false)

        localCallCounter = 0

        localQ = Calculus.gradient(
            p::Vector{Float64}->begin

            #return dgQ(p[1:dgAParamSize], p[dgAParamSize+1:end], probHandling, nDim, nGauss, nProbParams, dgAParamSize,  data)
            return dgQ(p[1:dgAParamSize], p[dgAParamSize+1:end], probHandling, nDim, nGauss, nProbParams, mA, dgAParamSize,  data)
        end)
        function localQGrad(p::Vector{Float64}, g::Vector{Float64})
            #println(p)
            callCounter += 1
            localCallCounter += 1
            nGrad = length(g)
            if nGrad>0
                grad = localQ(p)
                for i in 1:nGrad
                    g[i] = grad[i]
                end
                #println(g)
            end
            #setParameters(dgTry, p[1:dgAParamSize])

            #sl = dgQ(p[1:dgAParamSize], p[dgAParamSize+1:end], probHandling, nDim, nGauss, nProbParams, dgAParamSize,  data)
            sl = dgQ(p[1:dgAParamSize], p[dgAParamSize+1:end], probHandling, nDim, nGauss, nProbParams, mA, dgAParamSize,  data)
            if mod(callCounter, 500) == 0
                println("Call  #",callCounter, " local Call #", localCallCounter , " sl = ", sl)
            end
            #print(sl, "   ")

            if isnan(sl)
                println("***** sl == NaN *****")
                println("Call  #",callCounter, " sl = ", sl)
                println("dgTry: " , dgTry)
                println("p: ", p)
                println("g: ", g)
                #error("** error **")
            end

            return sl
        end

        fixedParams = startVector[1:dgAParamSize]
        localQHalf = Calculus.gradient(
            p::Vector{Float64}->begin

            #return dgQ(p[1:dgAParamSize], p[dgAParamSize+1:end], probHandling, nDim, nGauss, nProbParams, dgAParamSize,  data)
            return dgQ(fixedParams, p[1:dgAParamSize], probHandling, nDim, nGauss, nProbParams, mA, dgAParamSize,  data)
        end)
        function localQHalfGrad(p::Vector{Float64}, g::Vector{Float64})
            #println(p)
            callCounter += 1
            localCallCounter += 1
            nGrad = length(g)
            if nGrad>0
                grad = localQHalf(p)
                for i in 1:nGrad
                    g[i] = grad[i]
                end
                #println(g)
            end
            #setParameters(dgTry, p[1:dgAParamSize])

            #sl = dgQ(p[1:dgAParamSize], p[dgAParamSize+1:end], probHandling, nDim, nGauss, nProbParams, dgAParamSize,  data)
            sl = dgQ(fixedParams, p[1:dgAParamSize], probHandling, nDim, nGauss, nProbParams, mA, dgAParamSize,  data)
            if mod(callCounter, 500) == 0
                println("Call  #",callCounter, " local Call #", localCallCounter , " sl = ", sl)
            end
            #print(sl, "   ")

            if isnan(sl)
                println("***** sl == NaN *****")
                println("Call  #",callCounter, " sl = ", sl)
                println("dgTry: " , dgTry)
                println("p: ", p)
                println("g: ", g)
                #error("** error **")
            end

            return sl
        end
        if testBoundaries(startVector,lowerBounds,upperBounds) != length(startVector)
            return (lowerBounds, upperBounds, startVector, 9e99, startVector, :FORCED_STOP )
        end
        if fullOptimization
            optLocal = Opt(algo, length(startVector))
            lower_bounds!(optLocal, lowerBounds)
            upper_bounds!(optLocal, upperBounds)
            min_objective!(optLocal, localQGrad)
            #ftol_abs!(optLocal,fTolerance)
            ftol_rel!(optLocal,fTolerance)
            #xtol_rel!(optLocal,fTolerance)
            maxeval!(optLocal, callLimit)
            (minf, minx, ret) = optimize(optLocal,startVector)
        else
            optLocal = Opt(algo, div(length(startVector),2))
            lower_bounds!(optLocal, lowerBounds[dgAParamSize+1:end])
            upper_bounds!(optLocal, upperBounds[dgAParamSize+1:end])
            min_objective!(optLocal, localQHalfGrad)
            #ftol_abs!(optLocal,fTolerance)
            ftol_rel!(optLocal,fTolerance)
            #xtol_rel!(optLocal,fTolerance)
            maxeval!(optLocal, callLimit)
            (minf, minxHalf, ret) = optimize(optLocal,startVector[dgAParamSize+1:end])
            minx = vcat(startVector[1:dgAParamSize],minxHalf)
        end
        if verbose
            println(algo)
            println("startValues")
            println(startVector)
            println("lowerBounds")
            println(lowerBounds)
            println("upperBounds")
            println(upperBounds)
            println("minf")
            println(minf)
            println("minx")
            println(minx)
            println("ret")
            println(ret)
        end
       return (lowerBounds, upperBounds, startVector, minf, minx, ret)
    end



    lowerBounds = zeros(dgParamSize)
    upperBounds = zeros(dgParamSize)
    DlowerBounds = zeros(dgParamSize)
    DupperBounds = zeros(dgParamSize)
    #startValues = zeros(2*dgAParamSize)

    amplLowerBound = 0.0
    amplUpperBound = 0.0

    smallDx = 1e-30
    largeDx = 1.0
    startDx = -30.#-25.#1e-10
    sigmaMinimum = 0.01
    if probHandling == SqrtAmplitude
        if nGauss == 1
            amplLowerBound = 1.0
        else
            amplLowerBound = -1.0
        end
        amplUpperBound = 1.0
    elseif probHandling == AngleAmplitude
        amplLowerBound = -2*π
        amplUpperBound = 2*π
    end

    lowerBounds[1:nDim] .= -20.
    upperBounds[1:nDim] .= +20.
    lowerBounds[nDim+1:end] .= 1e-10#smallDx #0.01#smallDx
    upperBounds[nDim+1:end] .= +20000.

    DlowerBounds[1:end] .= -37.#smallDx
#    DupperBounds[1:end] .= -0. #largeDx
    DupperBounds[1:end] .= 37. #largeDx

    lbounds = vcat(repeat([amplLowerBound],nProbParams), repeat(lowerBounds, nGauss))
    ubounds = vcat(repeat([amplUpperBound],nProbParams), repeat(upperBounds, nGauss))
    dlbounds = fill(-137., dgAParamSize)
    dubounds = fill(0., dgAParamSize) #bookmark

    startDxValues = fill(startDx, dgAParamSize) #vcat(startDx, fill(startDx, nDim*2))


    randomStart = Float64[]
    if startParameters == nothing
        if false
            #copied from old code, check is needed
            randomStart = randomGaussStartingPoint(nDim, nGauss, 0.0, 1.0, 1.5, 0.3,sigmaMinimum, true )
            while testBoundaries(randomStart, lbounds, ubounds) != length(randomStart)
                randomStart = randomGaussStartingPoint(nDim, nGauss, 0.0, 1.0, 1.5, 0.3,sigmaMinimum, true )
            end
        else
            #randomAngles = rand(nGauss-1)*2*π
            randomAmplitudes = rand(nGauss)
            randomAngles = sqrtAmplitudeToAngles(randomAmplitudes)
            if probHandling == SqrtAmplitude
                append!(randomStart, randomAmplitudes)
            elseif probHandling == AngleAmplitude
                append!(randomStart, randomAngles)
            end

            stdSample = std(data, dims=(2))
            subSample = data[:,sample(1:nData, nGauss, replace=false)]
            for i in 1:nGauss
                append!(randomStart, vcat(subSample[:, i], stdSample  ))
            end
        end

        startValues = vcat(randomStart,startDxValues)

    else
        startValues = vcat(startParameters[1:dgAParamSize],startDxValues)
    end


    lBoundsFull = vcat(lbounds, dlbounds)
    uBoundsFull = vcat(ubounds, dubounds)
    (lv1, uv1, sv1, minfCOBYLA, minxCOBYLA, retCOBYLA) = OptimizeMe(:LN_SBPLX, startValues, lBoundsFull, uBoundsFull, fTolerance, callLimits[1], fullOpt)


    #    LD_MMA
    #    LD_LBFGS

    startValues2 = deepcopy(minxCOBYLA)
    lBoundsFull2 = deepcopy(lBoundsFull)
    uBoundsFull2 = deepcopy(uBoundsFull)

    # setParameters(dgTry, startValues2[1:dgAParamSize])
    # #dgSumLogConservativeNoColons(dgTry, startValues2[dgAParamSize+1:end], data, true, true)
    # dgSumLogCDFTotalDxVary(dgTry, startValues2[dgAParamSize+1:end], data, true, false, varyDx)

    println(retCOBYLA)
    if length(callLimits) == 1

        return (lBoundsFull, uBoundsFull, startValues2, minfCOBYLA, minxCOBYLA, retCOBYLA)
    end

    # if dScale > 1.0
    #     reScale = (0.95-dScale)/nDim/nData
    #     #startValues2[dgAParamSize+1:dgAParamSize*2] .-= log(dScale)/nDim -0.05
    #     startValues2[dgAParamSize+1:dgAParamSize*2].+= reScale
    # end
    if callLimits[2]!=0
        (lv2, uv2, sv2, minfCOBYLA2, minxCOBYLA2, retCOBYLA2) = OptimizeMe(:GN_ESCH, startValues2, lBoundsFull2, uBoundsFull2, fTolerance, callLimits[2])
    else
        (lv2, uv2, sv2, minfCOBYLA2, minxCOBYLA2, retCOBYLA2) = (lBoundsFull, uBoundsFull, startValues2, minfCOBYLA, minxCOBYLA, retCOBYLA)
    end

    println(retCOBYLA2)
    if length(callLimits) == 2
    #        println(retCOBYLA2)
        return (lBoundsFull, uBoundsFull, startValues2, minfCOBYLA2, minxCOBYLA2, retCOBYLA2)
    end

    startValues3 = deepcopy(minxCOBYLA2)
    lBoundsFull3 = deepcopy(lBoundsFull)
    uBoundsFull3 = deepcopy(uBoundsFull)

    (lv3, uv3, sv3, minfCOBYLA3, minxCOBYLA3, retCOBYLA3) = OptimizeMe(:LN_SBPLX, startValues3, lBoundsFull3, uBoundsFull3, fTolerance, callLimits[3])

    println(retCOBYLA3)
    if length(callLimits) == 3
    #        println(retCOBYLA3)
        return (lBoundsFull, uBoundsFull, minxCOBYLA2, minfCOBYLA3, minxCOBYLA3, retCOBYLA3)
    end

    startValues4 = deepcopy(minxCOBYLA3)
    lBoundsFull4 = deepcopy(lBoundsFull)
    uBoundsFull4 = deepcopy(uBoundsFull)

    (lv4, uv4, sv4,minfBFGS, minxBFGS, retBFGS) = OptimizeMe(:LD_MMA, startValues4, lBoundsFull4, uBoundsFull4, fTolerance, callLimits[4] )

    println(retBFGS)
    return (lBoundsFull, uBoundsFull, minxCOBYLA3, minfBFGS, minxBFGS, retBFGS)

end

function SumLogOptimizer(data::Array{Float64,2}, nGauss::Int, callLimits::Vector{Int} = [1000,1000])
    nDim = size(data)[1]
    dgTry = randomGaussStartingPoint(nDim, nGauss, 0.0, 1.0, 1.0)

    nData = size(data)[2]
    dgParamSize = (2*nDim+1)
    dgAParamSize = dgParamSize*nGauss
    linearLogApprox = false
    nDxParams = 0
    varyDx = false
    if varyDx
        nDxParams = nDim
    end

    fTolerance = 1e-10


    callCounter = 0


    function OptimizeMe(algo::Symbol,  startVector::Vector{Float64}, lowerBounds::Vector{Float64}, upperBounds::Vector{Float64}, fTolerance::Float64, callLimit::Int64, verbose = false)

        localSumLogGrad = Calculus.gradient(
            p::Vector{Float64}->begin
            setParameters(dgTry, p[1:dgAParamSize])
            #return dgSumLogConservativeNoColons(dgTry, p[dgAParamSize+1:end], data, true, linearLogApprox)
            #return dgSumLogCDFFullDxVary(dgTry, p[dgAParamSize+1:end], data, true, linearLogApprox,varyDx)
            return dgSumLogCDFTotalDxVary(dgTry, p[dgAParamSize+1:end], data, true, linearLogApprox,varyDx)
        end)
        localCallCounter = 0
        function dgLocalSumLogAndGrad(p::Vector{Float64}, g::Vector{Float64})
            #println(p)
            callCounter += 1
            localCallCounter += 1
            nGrad = length(g)
            if nGrad>0
                grad = localSumLogGrad(p)
                for i in 1:nGrad
                    g[i] = grad[i]
                end
                #println(g)
            end
            setParameters(dgTry, p[1:dgAParamSize])
            #sl = dgSumLogCDFFullDxVary(dgTry,  p[dgAParamSize+1:end], data, true, linearLogApprox,varyDx)
            sl = dgSumLogCDFTotalDxVary(dgTry,  p[dgAParamSize+1:end], data, true, linearLogApprox,varyDx)
            if mod(callCounter, 100) == 0
                println("Call  #",callCounter, " local Call #", localCallCounter , " sl = ", sl)
            end
            #print(sl, "   ")

            if isnan(sl)
                println("***** sl == NaN *****")
                println("Call  #",callCounter, " sl = ", sl)
                println("dgTry: " , dgTry)
                println("p: ", p)
                println("g: ", g)
                #error("** error **")
            end

            return sl
        end

        optLocal = Opt(algo, length(startVector))
        lower_bounds!(optLocal, lowerBounds)
        upper_bounds!(optLocal, upperBounds)
        min_objective!(optLocal, dgLocalSumLogAndGrad)
        ftol_rel!(optLocal,fTolerance)
        maxeval!(optLocal, callLimit)
        (minf, minx, ret) = optimize(optLocal,startVector)

        if verbose
            println(algo)
            println("startValues")
            println(startVector)
            println("lowerBounds")
            println(lowerBounds)
            println("upperBounds")
            println(upperBounds)
            println("minf")
            println(minf)
            println("minx")
            println(minx)
            println("ret")
            println(ret)
        end
        return return (lowerBounds, upperBounds, startVector, minf, minx, ret)
    end


    lowerBounds = zeros(dgParamSize)
    upperBounds = zeros(dgParamSize)
    DlowerBounds = zeros(dgParamSize)
    DupperBounds = zeros(dgParamSize)
    startValues = zeros(dgAParamSize*2+nDxParams)
    Dxlowerbounds = zeros(nDim)
    Dxupperbounds = zeros(nDim)
    Dxstartvalues = zeros(nDim)

    smallDx = 1e-30
    largeDx = 1.0
    startDx = -10.#-25.#1e-10
    sigmaMinimum = 0.01
    if nGauss == 1
        lowerBounds[1] = 1.0
    else
#        lowerBounds[1] = smallDx
        lowerBounds[1] = -1.0
    end
    upperBounds[1] = 1.0

    lowerBounds[2:nDim+1] .= -10.
    upperBounds[2:nDim+1] .= +10.
    lowerBounds[nDim+2:end] .= smallDx #0.01#smallDx
    upperBounds[nDim+2:end] .= +200.

    DlowerBounds[1:end] .= -137.#smallDx
    DupperBounds[1] = -0.
    DupperBounds[2:end] .= -0. #largeDx

    Dxlowerbounds .= -20.
    Dxupperbounds .= +20.
    Dxstartvalues .= 0.0


    lbounds = repeat(lowerBounds, nGauss)
    ubounds = repeat(upperBounds, nGauss)
    dlbounds = repeat(DlowerBounds, nGauss)
    dubounds = repeat(DupperBounds, nGauss)

    startDxValues = vcat(startDx, fill(startDx, nDim*2))

    randomStart = Float64[]
    if false
        randomStart = randomGaussStartingPoint(nDim, nGauss, 0.0, 1.0, 1.5, 0.3,sigmaMinimum, true )
        while testBoundaries(randomStart, lbounds, ubounds) != length(randomStart)
            randomStart = randomGaussStartingPoint(nDim, nGauss, 0.0, 1.0, 1.5, 0.3,sigmaMinimum, true )
        end
    else
        stdSample = std(data, dims=(2))
        subSample = data[:,sample(1:nData, nGauss, replace=false)]
        for i in 1:nGauss
            append!(randomStart, vcat(rand(1),subSample[:, i], stdSample  ))
        end
    end
#    return randomStart

    startValues[1:2*dgAParamSize] = vcat(randomStart,
                        repeat(startDxValues, nGauss)
                        #,Dxstartvalues
                        )
                        #fill(startDx, dgAParamSize))
                        #startDx*rand(dgAParamSize))


    lBoundsFull = vcat(lbounds, dlbounds)
    uBoundsFull = vcat(ubounds, dubounds)

    if varyDx
        startValues = vcat(startValues, Dxstartvalues)
        lBoundsFull = vcat(lBoundsFull, Dxlowerbounds)
        uBoundsFull = vcat(uBoundsFull, Dxupperbounds)
    end

    (lv1, uv1, sv1, minfCOBYLA, minxCOBYLA, retCOBYLA) = OptimizeMe(:LN_SBPLX, startValues, lBoundsFull, uBoundsFull, fTolerance, callLimits[1])


#    LD_MMA
#    LD_LBFGS

    startValues2 = deepcopy(minxCOBYLA)
    lBoundsFull2 = deepcopy(lBoundsFull)
    uBoundsFull2 = deepcopy(uBoundsFull)

    # setParameters(dgTry, startValues2[1:dgAParamSize])
    # #dgSumLogConservativeNoColons(dgTry, startValues2[dgAParamSize+1:end], data, true, true)
    # dgSumLogCDFTotalDxVary(dgTry, startValues2[dgAParamSize+1:end], data, true, false, varyDx)

    println(retCOBYLA)
    if length(callLimits) == 1

        return (lBoundsFull, uBoundsFull, startValues2, minfCOBYLA, minxCOBYLA, retCOBYLA)
    end

    # if dScale > 1.0
    #     reScale = (0.95-dScale)/nDim/nData
    #     #startValues2[dgAParamSize+1:dgAParamSize*2] .-= log(dScale)/nDim -0.05
    #     startValues2[dgAParamSize+1:dgAParamSize*2].+= reScale
    # end

    (lv2, uv2, sv2, minfCOBYLA2, minxCOBYLA2, retCOBYLA2) = OptimizeMe(:GN_ESCH, startValues2, lBoundsFull2, uBoundsFull2, fTolerance, callLimits[2])

    println(retCOBYLA2)
    if length(callLimits) == 2
#        println(retCOBYLA2)
        return (lBoundsFull, uBoundsFull, startValues2, minfCOBYLA2, minxCOBYLA2, retCOBYLA2)
    end

    startValues3 = deepcopy(minxCOBYLA2)
    lBoundsFull3 = deepcopy(lBoundsFull)
    uBoundsFull3 = deepcopy(uBoundsFull)

    (lv3, uv3, sv3, minfCOBYLA3, minxCOBYLA3, retCOBYLA3) = OptimizeMe(:LN_SBPLX, startValues3, lBoundsFull3, uBoundsFull3, fTolerance, callLimits[3])

    println(retCOBYLA3)
    if length(callLimits) == 3
#        println(retCOBYLA3)
        return (lBoundsFull, uBoundsFull, minxCOBYLA2, minfCOBYLA3, minxCOBYLA3, retCOBYLA3)
    end

    startValues4 = deepcopy(minxCOBYLA3)
    lBoundsFull4 = deepcopy(lBoundsFull)
    uBoundsFull4 = deepcopy(uBoundsFull)

    (lv4, uv4, sv4,minfBFGS, minxBFGS, retBFGS) = OptimizeMe(:LD_MMA, startValues4, lBoundsFull4, uBoundsFull4, fTolerance, callLimits[4] )

    println(retBFGS)
    return (lBoundsFull, uBoundsFull, minxCOBYLA3, minfBFGS, minxBFGS, retBFGS)

end

function QReoptimizer(data::Array{Float64,2}, startVector::Vector{Float64}, probHandling::ProbabilityParameterisation, callLimit = 1000, algo::Symbol = :LN_COBYLA)
    nDim =size(data)[1]
    dgParamSize = 2*nDim
    if probHandling == SqrtAmplitude
        nProbParams = nGauss

    else
        if probHandling == AngleAmplitude
            nProbParams = nGauss -1
        end
    end
    dgAParamSize = dgParamSize*nGauss + nProbParams

    fTolerance = 1e-10
    callCounter = 0
    mA = marginalCDFDerivativeAllocations{Float64}(nDim, nGauss, nProbParams)

    function OptimizeMe(algo::Symbol,
                startVector::Vector{Float64},
                lowerBounds::Vector{Float64},
                upperBounds::Vector{Float64},
                fTolerance::Float64, callLimit::Int64, verbose = false)

        localQ = Calculus.gradient(
            p::Vector{Float64}->begin

            #return dgQ(p[1:dgAParamSize], p[dgAParamSize+1:end], probHandling, nDim, nGauss, nProbParams, dgAParamSize,  data)
            return dgQ(p[1:dgAParamSize], p[dgAParamSize+1:end], probHandling, nDim, nGauss, nProbParams, mA, dgAParamSize,  data)
        end)
        localCallCounter = 0
        function localQGrad(p::Vector{Float64}, g::Vector{Float64})
            #println(p)
            callCounter += 1
            localCallCounter += 1
            nGrad = length(g)
            if nGrad>0
                grad = localQ(p)
                for i in 1:nGrad
                    g[i] = grad[i]
                end
                #println(g)
            end
            #setParameters(dgTry, p[1:dgAParamSize])

            #sl = dgQ(p[1:dgAParamSize], p[dgAParamSize+1:end], probHandling, nDim, nGauss, nProbParams, dgAParamSize,  data)
            sl = dgQ(p[1:dgAParamSize], p[dgAParamSize+1:end], probHandling, nDim, nGauss, nProbParams, mA, dgAParamSize,  data)
            if mod(callCounter, 100) == 0
                println("Call  #",callCounter, " local Call #", localCallCounter , " sl = ", sl)
            end
            #print(sl, "   ")

            if isnan(sl)
                println("***** sl == NaN *****")
                println("Call  #",callCounter, " sl = ", sl)
                println("dgTry: " , dgTry)
                println("p: ", p)
                println("g: ", g)
                #error("** error **")
            end

            return sl
        end
        optLocal = Opt(algo, length(startVector))
        lower_bounds!(optLocal, lowerBounds)
        upper_bounds!(optLocal, upperBounds)
        min_objective!(optLocal, localQGrad)
        ftol_rel!(optLocal,fTolerance)
        maxeval!(optLocal, callLimit)
        (minf, minx, ret) = optimize(optLocal,startVector)

        if verbose
            println(algo)
            println("startValues")
            println(startVector)
            println("lowerBounds")
            println(lowerBounds)
            println("upperBounds")
            println(upperBounds)
            println("minf")
            println(minf)
            println("minx")
            println(minx)
            println("ret")
            println(ret)
        end
        return return (lowerBounds, upperBounds, startVector, minf, minx, ret)
    end
#unfinished
end


function SumLogReOptimizer(data::Array{Float64,2}, params::Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,1},Float64,Array{Float64,1},Symbol}, callLimit = 1000, algo::Symbol = :LN_COBYLA )
    nDim = size(data)[1]
    dgParamSize = (2*nDim+1)
    nGauss = div(length(params[5]),2*dgParamSize)
    #println( length(params[5])," ",2*dgParamSize )
    dgAParamSize = dgParamSize*nGauss
    linearLogApprox = false
    nDxParams = 0

    varyDx = false
    if varyDx
        nDxParams = nDim
    end

    dgTry = dgAConstruct(params[5][1:dgAParamSize], nDim)
    #dgLocal = dgAConstruct(params[5][1:dgAParamSize], nDim)

    callCounter = 0
    fTolerance = 1e-30

    function OptimizeMe(algo::Symbol,  startVector::Vector{Float64}, lowerBounds::Vector{Float64}, upperBounds::Vector{Float64}, fTolerance::Float64, callLimit::Int64, verbose = false)

        localSumLogGrad = Calculus.gradient(
            p::Vector{Float64}->begin
            setParameters(dgTry, p[1:dgAParamSize])
            #return dgSumLogConservativeNoColons(dgTry, p[dgAParamSize+1:end], data, true, linearLogApprox)
            #return dgSumLogCDFFullDxVary(dgTry, p[dgAParamSize+1:end], data, true, linearLogApprox,varyDx)
            return dgSumLogCDFTotalDxVary(dgTry, p[dgAParamSize+1:end], data, true, linearLogApprox,varyDx)
        end)
        localCallCounter = 0
        function dgLocalSumLogAndGrad(p::Vector{Float64}, g::Vector{Float64})
            #println(p)
            callCounter += 1
            localCallCounter += 1
            nGrad = length(g)
            if nGrad>0
                grad = localSumLogGrad(p)
                for i in 1:nGrad
                    g[i] = grad[i]
                end
                #println(g)
            end
            setParameters(dgTry, p[1:dgAParamSize])
            #sl = dgSumLogCDFFullDxVary(dgTry,  p[dgAParamSize+1:end], data, true, linearLogApprox,varyDx)
            sl = dgSumLogCDFTotalDxVary(dgTry,  p[dgAParamSize+1:end], data, true, linearLogApprox,varyDx)
            if mod(callCounter, 100) == 0
                println("Call  #",callCounter, " local Call #", localCallCounter , " sl = ", sl)
            end
            #print(sl, "   ")

            if isnan(sl)
                println("***** sl == NaN *****")
                println("Call  #",callCounter, " sl = ", sl)
                println("dgTry: " , dgTry)
                println("p: ", p)
                println("g: ", g)
                #error("** error **")
            end

            return sl
        end

        optLocal = Opt(algo, length(startVector))
        lower_bounds!(optLocal, lowerBounds)
        upper_bounds!(optLocal, upperBounds)
        min_objective!(optLocal, dgLocalSumLogAndGrad)
        ftol_rel!(optLocal,fTolerance)
        maxeval!(optLocal, callLimit)
        (minf, minx, ret) = optimize(optLocal,startVector)

        if verbose
            println(algo)
            println("startValues")
            println(startVector)
            println("lowerBounds")
            println(lowerBounds)
            println("upperBounds")
            println(upperBounds)
            println("minf")
            println(minf)
            println("minx")
            println(minx)
            println("ret")
            println(ret)
        end
        return return (lowerBounds, upperBounds, startVector, minf, minx, ret)
    end


    (lv2, uv2, sv2, minf, minx, ret) = OptimizeMe(algo, params[5], params[1], params[2], fTolerance, callLimit )


    #println(typeof(params[1]), " ", length(params[1]), " " , 2*dgAParamSize)
    # opt = Opt(algo, 2*dgAParamSize+nDxParams)
    # opt.vector_storage = 1
    # lower_bounds!(opt, params[1])
    # upper_bounds!(opt, params[2])
    # min_objective!(opt, dgLocalSumLogAndGrad)
    # ftol_rel!(opt, fTolerance)
    # maxeval!(opt, callLimit)
    # println("x")
    # (minf, minx, ret) = optimize(opt, params[5])
    #
    # println( localSumLogGrad(params[5]) )

    # println(minx)
    println(minf)
    println(ret)
    #return (params[1], params[2], params[5], minf, minx, ret)
    return (lv2, uv2, sv2, minf, minx, ret)

end


zSmall = deepcopy(z3[:,1:1000])
zSmall100 = deepcopy(z3[:,1:100])
zSmall200 = deepcopy(z3[:,1:200])

precompile(SumLogOptimizer, (Array{Float64,2}, Int64, Vector{Int64}))
precompile(dgSumLogCDF,(Array{diagonalGauss{Float64},1}, Vector{Float64}, Array{Float64,2}, Bool, Bool, Bool))
# precompile(dgMarginalCDFDerivative,(Array{diagonalGauss{Float64},1},
#     Vector{Float64},Union{Nothing, Array{Float64, 2}},
#     Union{Nothing, Array{Float64, 3}},
#     Union{Nothing, Array{Float64, 3}}, Bool))

precompile(dgMarginalCDFDerivative,(Array{diagonalGauss{Float64},1},
        Vector{Float64},Array{Float64, 2},
        Array{Float64, 3},
        Array{Float64, 3},
        Array{Float64, 1}, Bool))

precompile(SumLogReOptimizer,(Array{Float64,2},Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,1},Float64,Array{Float64,1},Symbol}, Int64, Symbol ))

precompile(resetMarginalCDFAllocations,(marginalCDFDerivativeAllocations{Float64},))

precompile(dgAConstructSwitch,(Array{Float64,1},
            Int64,
            ProbabilityParameterisation,
            Union{Nothing,Vector{Float64}},
            Union{Nothing,Array{Float64,2}}))

precompile(dgAConstructSwitch,(Array{Float64,1},
                        Int64,
                        ProbabilityParameterisation,
                        Vector{Float64},
                        Array{Float64,2}))

precompile(diagonalGauss, (Array{Float64,1},))
precompile(marginalCDFDerivativeAllocations{Float64}, (Int64,Int64,Int64))

precompile(dgExternalAmplitudesMarginalCDFDerivative,(Array{diagonalGauss{Float64},1},
        Vector{Float64},
        Array{Float64,2},
        Vector{Float64},
        Array{Float64,2},
        Array{Float64,3},
        Array{Float64,3},
        Array{Float64,1},
        Bool))

precompile(dgQ, (Vector{Float64},Vector{Float64},ProbabilityParameterisation, Int64, Int64, Int64, marginalCDFDerivativeAllocations{Float64}, Int64, Array{Float64,2} ) )

precompile(QOptimizer, ( Array{Float64,2}, Int64, Vector{Int64}, ProbabilityParameterisation, Nothing))
precompile(QOptimizer, ( Array{Float64,2}, Int64, Vector{Int64}, ProbabilityParameterisation, Vector{Float64}))

resultArray1 = Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,1},Float64,Array{Float64,1},Symbol}[]
resultArray2 = Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,1},Float64,Array{Float64,1},Symbol}[]
resultArray3 = Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,1},Float64,Array{Float64,1},Symbol}[]
resultArray4 = Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,1},Float64,Array{Float64,1},Symbol}[]
resultArray5 = Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,1},Float64,Array{Float64,1},Symbol}[]
resultArray6 = Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,1},Float64,Array{Float64,1},Symbol}[]
resultArray7 = Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,1},Float64,Array{Float64,1},Symbol}[]

#push!(resultArray3, QOptimizer(zSmall100, 5, [1000,1000,2000]))
# push!(resultArray3,SumLogOptimizer(zSmall,5, [1000,1000,2000]))
# push!(resultArray3,SumLogOptimizer(zSmall100,3, [1000, 15000, 35000]))
# push!(resultArray3,SumLogOptimizer(zSmall,9, [10000,1000, 20000, 500]))
# scatter([(i[1],i[2]) for i in eachcol(zSmall)]); plotGaussian(resultArray3[end][5], 2)
#
# refitTestParams = deepcopy(resultArray4[end])
# refitTestParams = deepcopy(resultArray3[end])
#
# push!(resultArray4, SumLogReOptimizer(zSmall200, refitTestParams, 20000, :LN_COBYLA ))
# push!(resultArray4, SumLogReOptimizer(zSmall200, refitTestParams, 1000, :LD_MMA ))
# push!(resultArray4, SumLogReOptimizer(zSmall100, refitTestParams, 1000, :LD_LBFGS ))
#
# scatter([(i[1],i[2]) for i in eachcol(zSmall)]); plotGaussian(resultArray4[end][5], 2,7)
#
#
# resultArrayTest = Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,1},Float64,Array{Float64,1},Symbol}[]
# v3Small100 = [ rand(d3,100) for i in 1:100]
# push!(resultArrayTest,SumLogOptimizer(v3Small100[3],5, [10000,1000, 20000, 500]))
# scatter([(i[1],i[2]) for i in eachcol(v3Small100[3])]); plotGaussian(resultArrayTest[end][5], 2)
#
# scatter([(i[1],i[2]) for i in eachcol(v3Small100[4])]); plotGaussian(repeatedTest[end-3][5][5], 2)
#
#
#
# srepeatedTest = []
# for sampleIndex in 1:10
# #sampleIndex
#     for j in 1:40
#         results = SumLogOptimizer(v3Small100[sampleIndex],7, [10000,1000, 20000, 500])
#         marker = length(srepeatedTest)+1
#         push!(srepeatedTest, ( sampleIndex, marker ,results[4], results[6], results ) )
#     end
# end
#
# @save "samplesAndRuns.jld2" v3Small100 srepeatedTest
#
# histogram([r[3] for r in filter(x->x[4]==:FTOL_REACHED || x[4]==:XTOL_REACHED, srepeatedTest)])
# plotSample =1; scatter([(i[1],i[2]) for i in eachcol(v3Small100[plotSample])]); plotGaussian(srepeatedTest[findMinimum(srepeatedTest,plotSample)][5][5], 2)
# #bela = dgPrune(srepeatedTest[2][5])
#
#
#
# scatter([(i[1],i[2]) for i in eachcol(v3Small100[1])]); plotGaussian(repeatedTest[findMinimum(repeatedTest, 1 )][5][5], 2)
# scatter([(i[1],i[2]) for i in eachcol(v3Small100[1])]); plotGaussian(srepeatedTest[30][5][5], 2)
# scatter([(i[1],i[2]) for i in eachcol(v3Small100[2])]); plotGaussian(bela[5], 2)
#
#
#
# srepeatedTest = []
# for i in 1:length(repeatedTest)
#     println(repeatedTest[i][1], "  ", i)
#     #push!(srepeatedTest, (repeatedTest[i][1], i, repeatedTest[i][3], repeatedTest[i][4], repeatedTest[i][5]))
# end

function findMinimum( inputArray::Array, sampleIndex )
    filteredArray = filter(x->x[1]==sampleIndex && (x[4]==:FTOL_REACHED || x[4]==:XTOL_REACHED) , inputArray)
    index = argmin([r[3] for r in filteredArray])
    constraindMinimumIndex = filteredArray[index][2]
    println(constraindMinimumIndex, " ", index)
    return constraindMinimumIndex
end

function dgPrune(inputResults::Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,1},Float64,Array{Float64,1},Symbol}, nDim = 2, amplitudeCut = 1e-10, sigmaCut = 1e-40)
    dgParamSize = (2*nDim+1)
    nGauss = div(length(inputResults[5]),2*dgParamSize)
    dgAParamSize = dgParamSize*nGauss




    amplitudes = inputResults[5][1:dgParamSize:dgAParamSize].^2
    nPrune = sum( x->x>amplitudeCut, amplitudes )
    dgBParamSize = dgParamSize*nPrune

    sv = zeros(dgBParamSize*2)
    lv = zeros(dgBParamSize*2)
    uv = zeros(dgBParamSize*2)
    minx = zeros(dgBParamSize*2)

    j = 0
    for i in 1:length(amplitudes)
        if amplitudes[i] > amplitudeCut
            #println(amplitudes[i])
            inputRange = range((i-1)*dgParamSize+1, length = dgParamSize)
            outputRange = range(j*dgParamSize+1, length = dgParamSize)
            sv[ outputRange ] = inputResults[3][inputRange]
            lv[ outputRange ] = inputResults[1][inputRange]
            uv[ outputRange ] = inputResults[2][inputRange]
            minx[ outputRange ] = inputResults[5][inputRange]
            j+=  1
            #println(minx[outputRange])
        end
    end
    j = 0
    for i in 1:length(amplitudes)
        if amplitudes[i] > amplitudeCut
            inputRange = range(dgAParamSize+  (i-1)*dgParamSize+1, length = dgParamSize)
            outputRange = range(dgBParamSize+ j*dgParamSize+1, length = dgParamSize)
            #println(i, " ", j, " ", dgBParamSize)
            #println(outputRange)
            sv[ outputRange ] = inputResults[3][inputRange]
            lv[ outputRange ] = inputResults[1][inputRange]
            uv[ outputRange ] = inputResults[2][inputRange]
            minx[ outputRange ] = inputResults[5][inputRange]
            j+=  1
        end
    end
 return (lv, uv, sv,inputResults[4] , minx, inputResults[6])
end

function dgPrune2old(inputResults::Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,1},Float64,Array{Float64,1},Symbol},
        nDim = 2,
        amplitudeCut = 1e-10,
        sigmaCut = 1e-40,
        returnDecisionOnly = false)

    dgParamSize = (2*nDim+1)
    nGauss = div(length(inputResults[5]),2*dgParamSize)
    dgAParamSize = dgParamSize*nGauss


    lowerBounds = zeros(dgParamSize)
    upperBounds = zeros(dgParamSize)
    lowerBounds[1] = sqrt(amplitudeCut)
    lowerBounds[2:nDim+1] .= 0.0
    lowerBounds[nDim+2:dgParamSize] .= sigmaCut
    upperBounds[1] = 1.0
    upperBounds[2:nDim+1] .= Inf
    upperBounds[nDim+2:dgParamSize] .= Inf

    lowerBoundsVector = repeat(lowerBounds, nGauss)
    upperBoundsVector = repeat(upperBounds, nGauss)

    decisionsVector = [(s<=u) && (s>=l) for (s,l,u) in zip(abs.(inputResults[5][1:dgAParamSize]), lowerBoundsVector, upperBoundsVector)]
    decisions = [ sum(r)==dgParamSize for r in Iterators.partition(decisionsVector,dgParamSize) ]

    # amplitudes = inputResults[5][1:dgParamSize:dgAParamSize].^2
    # nPrune = sum( x->x>amplitudeCut, amplitudes )
    nPrune = sum(decisions)
    if returnDecisionOnly
        return nPrune<nGauss
    end

    dgBParamSize = dgParamSize*nPrune

    sv = zeros(dgBParamSize*2)
    lv = zeros(dgBParamSize*2)
    uv = zeros(dgBParamSize*2)
    minx = zeros(dgBParamSize*2)


    j = 0
    for i in 1:nGauss
        if decisions[i]
            #println(amplitudes[i])
            inputRange = range((i-1)*dgParamSize+1, length = dgParamSize)
            outputRange = range(j*dgParamSize+1, length = dgParamSize)
            sv[ outputRange ] = inputResults[3][inputRange]
            lv[ outputRange ] = inputResults[1][inputRange]
            uv[ outputRange ] = inputResults[2][inputRange]
            minx[ outputRange ] = inputResults[5][inputRange]
            j+=  1
            #println(minx[outputRange])
        end
    end
    j = 0
    for i in 1:nGauss
        if decisions[i]
            inputRange = range(dgAParamSize+  (i-1)*dgParamSize+1, length = dgParamSize)
            outputRange = range(dgBParamSize+ j*dgParamSize+1, length = dgParamSize)
            #println(i, " ", j, " ", dgBParamSize)
            #println(outputRange)
            sv[ outputRange ] = inputResults[3][inputRange]
            lv[ outputRange ] = inputResults[1][inputRange]
            uv[ outputRange ] = inputResults[2][inputRange]
            minx[ outputRange ] = inputResults[5][inputRange]
            j+=  1
        end
    end

    if nPrune > 0
        sqrAmplitudeNorm = sum(x->x^2, minx[1:dgParamSize:dgBParamSize] )
        #sqrAmplitudeNorm = maximum( abs.(minx[1:dgParamSize:dgBParamSize] ) )
        if nPrune == 1
            sqrAmplitudeNorm *= 1.001 #workaround for NLOpt boundary checks
        end

        #minx[1:dgParamSize:dgBParamSize] = abs.(inputResults[5][1:dgParamSize:dgBParamSize]) ./ sqrAmplitudeNorm #sqrt(sqrAmplitudeNorm)
        #minx[1:dgParamSize:dgBParamSize] = abs.(minx[1:dgParamSize:dgBParamSize]) ./ sqrAmplitudeNorm #sqrt(sqrAmplitudeNorm)
        minx[1:dgParamSize:dgBParamSize] = abs.(minx[1:dgParamSize:dgBParamSize]) ./ sqrt(sqrAmplitudeNorm)
    end
 return (lv, uv, sv,inputResults[4] , minx, inputResults[6])
end

function dgPrune2(inputResults::Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,1},Float64,Array{Float64,1},Symbol},
        nDim = 2,
        amplitudeCut = 1e-10,
        sigmaCut = 1e-40,
        returnDecisionOnly = false)
    function convertNewToOld(newStyleArray::Vector{Float64}, probabilities::Vector{Float64})
        oldStyleArray = zeros(length(newStyleArray))
        nGauss = length(probabilities)
        dgParamSize = div(length(oldStyleArray), nGauss)
        oldStyleArray[1:dgParamSize:end] = probabilities
        for i in 1:dgParamSize-1
            oldStyleArray[i+1:dgParamSize:end] = newStyleArray[nGauss+i:dgParamSize-1:end]
        end
        return oldStyleArray
    end
    function convertOldToNew(oldStyleArray::Vector{Float64}, amplParams::Vector{Float64})
        newStyleArray = zeros(length(oldStyleArray))
        nGauss = length(amplParams)
        dgParamSize = div(length(oldStyleArray), nGauss)
        newStyleArray[1:nGauss] = amplParams
        for i in 1:dgParamSize-1
            newStyleArray[nGauss+i:dgParamSize-1:end] = oldStyleArray[i+1:dgParamSize:end]
        end
        return newStyleArray
    end


    dgParamSize = (2*nDim)+1
    nGauss = div(length(inputResults[5]),2*dgParamSize)
    dgAParamSize = dgParamSize*nGauss
    nProbParams = nGauss
    oldStyleParams = zeros(dgAParamSize*2)

    amplitudeParams = inputResults[5][1:nProbParams]
    probabilities = zeros(nGauss)
    probabilityGradient = zeros(nProbParams, nGauss)
    amplitudeToProbability(amplitudeParams, probabilities, probabilityGradient)
    amplitudes = sqrt.(probabilities)
    oldStyleParams[1:dgAParamSize] = convertNewToOld(inputResults[5][1:dgAParamSize], amplitudes)
    oldStyleParams[dgAParamSize+1:end] = convertNewToOld(inputResults[5][dgAParamSize+1:end], inputResults[5][dgAParamSize+1:dgAParamSize+nProbParams])

    lowerBounds = zeros(dgParamSize)
    upperBounds = zeros(dgParamSize)
    lowerBounds[1] = sqrt(amplitudeCut)
    lowerBounds[2:nDim+1] .= 0.0
    lowerBounds[nDim+2:dgParamSize] .= sigmaCut
    upperBounds[1] = 1.0
    upperBounds[2:nDim+1] .= Inf
    upperBounds[nDim+2:dgParamSize] .= Inf

    lowerBoundsVector = repeat(lowerBounds, nGauss)
    upperBoundsVector = repeat(upperBounds, nGauss)

    decisionsVector = [(s<=u) && (s>=l) for (s,l,u) in zip(abs.(oldStyleParams[1:dgAParamSize]), lowerBoundsVector, upperBoundsVector)]
    decisions = [ sum(r)==dgParamSize for r in Iterators.partition(decisionsVector,dgParamSize) ]

    # amplitudes = inputResults[5][1:dgParamSize:dgAParamSize].^2
    # nPrune = sum( x->x>amplitudeCut, amplitudes )
    nPrune = sum(decisions)
    if returnDecisionOnly
        return nPrune < nGauss
    end

    dgBParamSize = dgParamSize*nPrune

    # sv = zeros(dgBParamSize*2)
    # lv = zeros(dgBParamSize*2)
    # uv = zeros(dgBParamSize*2)
    minx = zeros(dgBParamSize*2)


    j = 0
    for i in 1:nGauss
        if decisions[i]
            #println(amplitudes[i])
            inputRange = range((i-1)*dgParamSize+1, length = dgParamSize)
            outputRange = range(j*dgParamSize+1, length = dgParamSize)
            # sv[ outputRange ] = inputResults[3][inputRange]
            # lv[ outputRange ] = inputResults[1][inputRange]
            # uv[ outputRange ] = inputResults[2][inputRange]
            minx[ outputRange ] = oldStyleParams[inputRange]
            j+=  1
            #println(minx[outputRange])
        end
    end
    j = 0
    for i in 1:nGauss
        if decisions[i]
            inputRange = range(dgAParamSize+  (i-1)*dgParamSize+1, length = dgParamSize)
            outputRange = range(dgBParamSize+ j*dgParamSize+1, length = dgParamSize)
            #println(i, " ", j, " ", dgBParamSize)
            #println(outputRange)
            # sv[ outputRange ] = inputResults[3][inputRange]
            # lv[ outputRange ] = inputResults[1][inputRange]
            # uv[ outputRange ] = inputResults[2][inputRange]
            minx[ outputRange ] = oldStyleParams[inputRange]
            j+=  1
        end
    end

    newAmplitudes = normalize(sqrt.(minx[1:dgParamSize:dgBParamSize]))
    newStyleMinX = zeros(dgBParamSize*2)
    newStyleMinX[1:dgBParamSize] = convertOldToNew(minx[1:dgBParamSize], newAmplitudes)
    nProbBParams = length(newAmplitudes)
#    println(minx[dgBParamSize+1:end], " " ,fill(-1.0, nProbParams))
    println(minx[dgBParamSize+1:dgBParamSize+nProbBParams])
    if nProbBParams!=0
        newStyleMinX[dgBParamSize+1:end] = convertOldToNew(minx[dgBParamSize+1:end], minx[dgBParamSize+1:dgBParamSize+nProbBParams])
    end

    return newStyleMinX
end


function dgPrune3(inputResults::Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,1},Float64,Array{Float64,1},Symbol},
        nDim = 2,
        amplitudeCut = 1e-10,
        sigmaCut = 1e-40,
        returnDecisionOnly = false)
    function convertNewToOld(newStyleArray::Vector{Float64}, probabilities::Vector{Float64})
        oldStyleArray = zeros(length(newStyleArray)+1)
        nGauss = length(probabilities)
        dgParamSize = div(length(oldStyleArray), nGauss)
        oldStyleArray[1:dgParamSize:end] = probabilities
        for i in 1:dgParamSize-1
            oldStyleArray[i+1:dgParamSize:end] = newStyleArray[nGauss-1+i:dgParamSize-1:end]
        end
        return oldStyleArray
    end
    function convertOldToNew(oldStyleArray::Vector{Float64}, angles::Vector{Float64})
        newStyleArray = zeros(length(oldStyleArray)-1)
        nGauss = length(angles)+1
        dgParamSize = div(length(oldStyleArray), nGauss)
        newStyleArray[1:nGauss-1] = angles
        for i in 1:dgParamSize-1
            newStyleArray[nGauss-1+i:dgParamSize-1:end] = oldStyleArray[i+1:dgParamSize:end]
        end
        return newStyleArray
    end


    dgParamSize = (2*nDim)+1
    nGauss = div(length(inputResults[5])+2,2*dgParamSize)
    dgAParamSize = dgParamSize*nGauss
    nProbParams = nGauss -1
    oldStyleParams = zeros(dgAParamSize*2)

    angles = inputResults[5][1:nProbParams]
    probabilities = zeros(nGauss)
    probabilityGradient = zeros(nProbParams, nGauss)
    angleToProbabilityAmplitudeUnpack(angles, probabilities, probabilityGradient)
    amplitudes = sqrt.(probabilities)
    oldStyleParams[1:dgAParamSize] = convertNewToOld(inputResults[5][1:dgAParamSize-1], amplitudes)
    oldStyleParams[dgAParamSize+1:end] = convertNewToOld(inputResults[5][dgAParamSize:end], fill(-1., nGauss)) # placeholder for angle delta parameters

    lowerBounds = zeros(dgParamSize)
    upperBounds = zeros(dgParamSize)
    lowerBounds[1] = sqrt(amplitudeCut)
    lowerBounds[2:nDim+1] .= 0.0
    lowerBounds[nDim+2:dgParamSize] .= sigmaCut
    upperBounds[1] = 1.0
    upperBounds[2:nDim+1] .= Inf
    upperBounds[nDim+2:dgParamSize] .= Inf

    lowerBoundsVector = repeat(lowerBounds, nGauss)
    upperBoundsVector = repeat(upperBounds, nGauss)

    decisionsVector = [(s<=u) && (s>=l) for (s,l,u) in zip(abs.(oldStyleParams[1:dgAParamSize]), lowerBoundsVector, upperBoundsVector)]
    decisions = [ sum(r)==dgParamSize for r in Iterators.partition(decisionsVector,dgParamSize) ]

    # amplitudes = inputResults[5][1:dgParamSize:dgAParamSize].^2
    # nPrune = sum( x->x>amplitudeCut, amplitudes )
    nPrune = sum(decisions)
    if returnDecisionOnly
        return nPrune < nGauss
    end

    dgBParamSize = dgParamSize*nPrune

    # sv = zeros(dgBParamSize*2)
    # lv = zeros(dgBParamSize*2)
    # uv = zeros(dgBParamSize*2)
    minx = zeros(dgBParamSize*2)


    j = 0
    for i in 1:nGauss
        if decisions[i]
            #println(amplitudes[i])
            inputRange = range((i-1)*dgParamSize+1, length = dgParamSize)
            outputRange = range(j*dgParamSize+1, length = dgParamSize)
            # sv[ outputRange ] = inputResults[3][inputRange]
            # lv[ outputRange ] = inputResults[1][inputRange]
            # uv[ outputRange ] = inputResults[2][inputRange]
            minx[ outputRange ] = oldStyleParams[inputRange]
            j+=  1
            #println(minx[outputRange])
        end
    end
    j = 0
    for i in 1:nGauss
        if decisions[i]
            inputRange = range(dgAParamSize+  (i-1)*dgParamSize+1, length = dgParamSize)
            outputRange = range(dgBParamSize+ j*dgParamSize+1, length = dgParamSize)
            #println(i, " ", j, " ", dgBParamSize)
            #println(outputRange)
            # sv[ outputRange ] = inputResults[3][inputRange]
            # lv[ outputRange ] = inputResults[1][inputRange]
            # uv[ outputRange ] = inputResults[2][inputRange]
            minx[ outputRange ] = oldStyleParams[inputRange]
            j+=  1
        end
    end

    newAngles = sqrtAmplitudeToAngles(minx[1:dgParamSize:dgBParamSize])
    newStyleMinX = zeros(dgBParamSize*2-2)
    newStyleMinX[1:dgBParamSize-1] = convertOldToNew(minx[1:dgBParamSize], newAngles)
    nProbBParams = length(newAngles)
#    println(minx[dgBParamSize+1:end], " " ,fill(-1.0, nProbParams))
    if nProbBParams!=0
        newStyleMinX[dgBParamSize:end] = convertOldToNew(minx[dgBParamSize+1:end], fill(-1.0, nProbBParams))
    else
        newStyleMinX[dgBParamSize:end] = minx[dgBParamSize+2:end]
    end

 return newStyleMinX
end

# scatter([(i[1],i[2]) for i in eachcol(v3Small100[2])]); plotGaussian(repeatedTest[argmax([ r[3] for r in repeatedTest ] ) ][5][5], 2) # scatter([(i[1],i[2]) for i in eachcol(v3Small100[1])]); plotGaussian(repeatedTest[findMinimum(repeatedTest, 1 )][5][5], 2)
# argmin([ r[3] for r in filter(x->x[1] == 2, repeatedTest) ]
#
# scatter([(i[1],i[2]) for i in eachcol(v3Small100[4])]); plotGaussian(srepeatedTest[217][5][5], 2)
# scatter([(i[1],i[2]) for i in eachcol(v3Small100[4])]); plotGaussian(srepeatedTest[105][5][5], 2)
#
# scatter([(i[1],i[2]) for i in eachcol(v3Small100[4])]); plotGaussian(bela2[5], 2)
#
# bela2 = dgPrune(srepeatedTest[217][5])
# bela5 = SumLogReOptimizer(v3Small100[4], bela4, 20000, :LD_MMA)
# scatter([(i[1],i[2]) for i in eachcol(v3Small100[4])]); plotGaussian(bela5[5], 2)

        # @time for i in 1:1
        #   push!(resultArray3, QOptimizer(z3, 7, [2000, 4000, 10000, 1000], AngleAmplitude))
        # end
        #
        # @profiler push!(resultArray3, QOptimizer(z3Small, 7, [2000, 4000, 10000, 1000], AngleAmplitude))
        #
        # pl = scatter( [(i[1],i[2]) for i in eachcol(z3)]); plotGaussianV3(resultArray3[end-1][5], 2,true, pl)
        # for res in resultArray3
        #     pl = scatter([(i[1],i[2]) for i in eachcol(zSmall100)]); plotGaussianV3(res[5], 2,true,pl,7)
        #     @printf("%3.3f %s \n", res[4], res[6])
        #     Plots.display(pl)
        #     wait_for_key(prompt="press key\n")
        # end


z1d1k = Array{Float64,2}(undef, (1, 1000))
for i in 1:1000
    z1d1k[1, i] = zSmall[1, i]
end

z1d10k = Array{Float64,2}(undef, (1, 10000))
for i in 1:10000
    z1d10k[1, i] = z3[1, i]
end

z1d = Array{Float64,2}(undef, (1, 100))
for i in 1:100
    z1d[1, i] = zSmall100[1, i]
end
z1d2 = Array{Float64,2}(undef, (1, 100))
for i in 1:100
    z1d2[1, i] = zSmall[1, i+100]
end
z1d50 = Array{Float64,2}(undef, (1, 50))
for i in 1:50
    z1d50[1, i] = zSmall[1, i+200]
end

function normalizeCrossValidation(inp::Array{Float64,2})
    (sizei,sizej) = size(inp)
    return [inp[i,j]<1e98 ? (inp[i,j]-inp[i,i])/inp[i,i] : Inf for i=1:sizei,j=1:sizej]
end

function printPercentages(inp::Array)
    return [(@sprintf "%3.1f%%\t" x*100) for x in inp ]
end

function printCrossValidationLatex(inp::Array{Float64,2})
    (sizei,sizej) = size(inp)
    println("\\begin{tabular}{",repeat(" r ",sizei),"}")
    strCV = Array{String,2}(undef,(sizei,sizej))
    strCV[1:sizei,1:sizej-1] .= [isfinite(x) ? (@sprintf "%3.1f\\%% &" x*100) : (@sprintf "\$\\infty\$ &") for x in inp[1:sizei,1:sizej-1]]
    strCV[1:sizei,sizej] .= [isfinite(x) ? (@sprintf "%3.1f\\%% \\\\" x*100) : (@sprintf "\$\\infty\$ \\\\" ) for x in inp[1:sizei,sizej]]
    for l in eachrow(strCV)
        println(l...)
    end
    println("\\end{tabular}")
end

function wait_for_key(; prompt = "press any key", io = stdin)
           setraw!(raw) = ccall(:jl_tty_set_mode, Int32, (Ptr{Cvoid},Int32), io.handle, raw)
           print(io, prompt)
           setraw!(true)
           read(io, 1)
           setraw!(false)
           nothing
end

function makeReport2d()

    global resultMatrix2dx5 = Array{Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,1},Float64,Array{Float64,1},Symbol},2}(undef,(10,33))

    Juno.progress() do id
        for i in 1:33
            for j in 6:10
                println("resultMatrix2d run start of ",j," ",i)
                resultMatrix2dx5[j,i] = QOptimizer(copy(z3[1:2,(j-1)*100+1:j*100]),5, [2000, 4000, 10000,1000],AngleAmplitude)
                println("resultMatrix2d run end   of ",j," ",i)
                @info "QOptimizer resultMatrix2d" progress=((i-1)*10 + j)/330 _id = id
            end
        end
    end
    #@load "resultMatrix2dx5-report-part1.jld2" resultMatrix2dx5 z3

    @save "resultMatrix2dx5-report-part1.jld2" resultMatrix2dx5 z3
    # prunedResultMatrix2d = Array{Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,1},Float64,Array{Float64,1},Symbol},2}(undef,(10,10))
    # Juno.progress() do id
    #     for i in 1:10
    #
    #         # myMin2 = findmin([r[4]+(r[4]<10.0)*1e99+(r[6]==:FORCED_STOP)*1e99 + (r[6]!=:XTOL_REACHED)*1e99 for r in resultMatrix2d[i,:]])
    #         #
    #         # println(myMin2)
    #         # myMin = myMin2[2]
    #         for j in 1:10
    #             println("prunedResultMatrix2d run start of ",j," ",i)
    #
    #             prunedResultMatrix2d[i,j] = QOptimizer(z3[1:2,(i-1)*100+1:i*100], 7, [2000, 4000, 10000, 1000], AngleAmplitude, dgPrune3(resultMatrix2d[i,j], 2, 1e-6, 1e-6) )
    #             println("prunedResultMatrix2d run end   of ",j," ",i)
    #             @info "QOptimizer prunedResultMatrix2d" progress=((i-1)*10 + j)/100 _id = id
    #         end
    #     end
    # end

    crossValidationMatrixPrunedQ2dx = zeros(10,10)
    crossValidationMatrixPrunedEntropy2dx = zeros(10,10)
    global prunedResultMatrix2dx5 = Array{Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,1},Float64,Array{Float64,1},Symbol},1}(undef,10)

    if false
        pyplot()
        for i=1:10
            sampleTitle = @sprintf "Sample #%d" i
            subSet = resultMatrix2dx5[i,:]
            myMin = findmin([r[4]+(r[6]==:FORCED_STOP)*1e99 + (r[6]!=:FTOL_REACHED  && r[6]!=:XTOL_REACHED )*1e99 + dgPrune3(r,2,1e-100, 1e-2, true)*1e99 for r in subSet])
            pl = plot()
            plotGaussianV3(subSet[myMin[2]][5], 2, false, pl )
            scatter!(pl,z3[1,(i-1)*100+1:i*100], z3[2,(i-1)*100+1:i*100], label = sampleTitle)
            Plots.display(pl)
            printGaussianParamsV4(subSet[myMin[2]][5],2,AngleAmplitude)

            println(subSet[myMin[2]][4], " " , subSet[myMin[2]][6])
            prunedResultMatrix2dx5[i] = QOptimizer(copy(z3[1:2,(i-1)*100+1:i*100]),5, [2000],AngleAmplitude,dgPrune3(subSet[myMin[2]],2,1e-6, 1e-2, false),false)
            printGaussianParamsV4(prunedResultMatrix2dx5[i][5],2,AngleAmplitude)
            filename = @sprintf "scatter2d on sample %d report2d-sigmacut-1e-10.eps" i
            savefig(pl, filename)
            wait_for_key(prompt="press key\n")
        end
    end

    for j in 1:10
    #    myMin = findmin(filter(x->x>10.0,[r[4] for r in resultMatrix[j,:]]))

        subSet = resultMatrix2dx5[j,:]
        myMin = findmin([r[4]+(r[6]==:FORCED_STOP)*1e99 + (r[6]!=:FTOL_REACHED  && r[6]!=:XTOL_REACHED )*1e99 + dgPrune3(r,2,1e-100, 1e-2, true)*1e99 for r in subSet])[2]



#        paramSet = subSet[myMin][5]
        paramSet = prunedResultMatrix2dx5[j][5]
        printGaussianParamsV4(paramSet,2,AngleAmplitude)
        dgAParamSize = div(length(paramSet) , 2)
        nGauss = div(dgAParamSize+1,5)
        nProbParams = nGauss  -1
        #println(length(subSet[myMin][5])," ", dgAParamSize, " ",nGauss," ", nProbParams)
        for i in 1:10
            mA = marginalCDFDerivativeAllocations{Float64}(2,nGauss,nProbParams)
            crossValidationMatrixPrunedQ2dx[j,i] = dgQ(paramSet[1:dgAParamSize],paramSet[dgAParamSize+1:2*dgAParamSize],AngleAmplitude,2, nGauss , nProbParams, mA, dgAParamSize, z3[1:2,(i-1)*100+1:i*100],false)

            crossValidationMatrixPrunedEntropy2dx[j,i] = entropyV3(paramSet, z3[1:2,(i-1)*100+1:i*100])
        end
    end
    @save "resultMatrix-report2d-part2.jld2" resultMatrix2dx5 z3 prunedResultMatrix2dx5 crossValidationMatrixPrunedEntropy2dx crossValidationMatrixPrunedQ2dx test2d10k test2d1k

    #test2d10k = QOptimizer(copy(z3[:,(1-1)*100+1:100*100]),5, [2000, 4000, 10000,1000],AngleAmplitude) #latest saved fmin = 34922.27056737154
    #test2d1k = QOptimizer(copy(z3[:,(1-1)*100+1:10*100]),5, [2000, 4000, 10000,1000],AngleAmplitude) #latest saved fmin = 3629.240596990492
    pl = plot(); plotGaussianV3(test2d10k[5],2,false,pl); scatter!(pl,z3[1,1:10000],z3[2,1:10000],label = "Sample 10k")
    savefig(pl, "scatter2d on 10k sample report2d-sigmacut-1e-10.eps")
    pl = plot(); plotGaussianV3(test2d1k[5],2,false,pl); scatter!(pl,z3[1,1:1000],z3[2,1:1000], label =  "Sample 1k")
    savefig(pl, "scatter2d on 1k sample report2d-sigmacut-1e-10.eps")

end

function makeReport()
    z1d10k = Array{Float64,2}(undef, (1, 10000))
    for i in 1:10000
        z1d10k[1, i] = z3[1, i]
    end
    #bookmark
    global resultMatrix53 = Array{Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,1},Float64,Array{Float64,1},Symbol},2}(undef,(10,100))


    Juno.progress() do id
        for i in 1:100
            for j in 1:10
                println("resultMatrix run start of ",j," ",i)
                resultMatrix53[j,i] = QOptimizer(copy(z1d10k[1:1,(j-1)*100+1:j*100]),5, [2000, 4000, 10000,1000],AngleAmplitude)
                #resultMatrix5s3[j,i] = QOptimizer(copy(z1d10k[1:1,(j-1)*100+1:j*100]),5, [2000, 4000, 10000,1000],SqrtAmplitude)
                println("resultMatrix run end   of ",j," ",i)
                @info "QOptimizer resultMatrix" progress=((i-1)*10 + j)/1000 _id = id
            end
        end
    end

    # global prunedResultMatrix = similar(resultMatrix) #Array{Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,1},Float64,Array{Float64,1},Symbol},2}(undef,(10,5))
    # Juno.progress() do id
    #     for i in 1:100
    #         # myMin2 = findmin([r[4]+(r[4]<10.0)*1e99+(r[6]==:FORCED_STOP)*1e99 for r in resultMatrix[i,:]])
    #         #
    #         # println(myMin2)
    #         # myMin = myMin2[2]
    #         for j in 1:1
    #             println("prunedResultMatrix run start of ",j," ",i)
    # #            prunedResultMatrix[i,j] = QOptimizer(z1d10k[1:1,(i-1)*100+1:i*100], 7, [2000, 4000, 10000, 1000], AngleAmplitude, dgPrune3(resultMatrix[i,myMin], 1, 1e-6, 1e-6) )
    #             #prunedResultMatrix[j,i] = QOptimizer(z1d10k[1:1,(j-1)*100+1:j*100], 7, [2000, 4000, 10000, 1000], AngleAmplitude, dgPrune3(resultMatrix[j,i], 1, 1e-6, 1e-6) )
    #             prunedResultMatrix[j,i] = brew(copy(z1d10k[1:1,(j-1)*100+1:j*100]), resultMatrix[j,i], 1e-4, 1e-5)
    #             println("prunedResultMatrix run end   of ",j," ",i)
    #             @info "QOptimizer prunedResultMatrix" progress=((i-1)*10 + j)/1000 _id = id
    #         end
    #     end
    # end
   #@load "resultMatrix-report-part1.jld2" resultMatrix53 z1d10k
    @save "resultMatrix-report-part1.jld2" resultMatrix53 z1d10k

    crossValidationMatrixPrunedQ = zeros(10,10)
    crossValidationMatrixPrunedEntropy = zeros(10,10)
    for j in 1:10
    #    myMin = findmin(filter(x->x>10.0,[r[4] for r in resultMatrix[j,:]]))

        subSet = resultMatrix53[j,:]
        myMin = findmin([r[4]+(r[6]==:FORCED_STOP)*1e99 + (r[6]!=:FTOL_REACHED  && r[6]!=:XTOL_REACHED )*1e99 + dgPrune3(r,1,1e-100, 1e-2, true)*1e99 for r in subSet])[2]
        dgAParamSize = div(length(subSet[myMin][5]) , 2)
        nGauss = div(dgAParamSize+1,3)
        nProbParams = nGauss  -1
        mA = marginalCDFDerivativeAllocations{Float64}(1,nGauss,nProbParams)
        #println(length(subSet[myMin][5])," ", dgAParamSize, " ",nGauss," ", nProbParams)
        for i in 1:10
            crossValidationMatrixPrunedQ[j,i] = dgQ(subSet[myMin][5][1:dgAParamSize],subSet[myMin][5][dgAParamSize+1:2*dgAParamSize],AngleAmplitude,1, nGauss , nProbParams, mA, dgAParamSize, z1d10k[1:1,(i-1)*100+1:i*100])

            crossValidationMatrixPrunedEntropy[j,i] = entropyV3(subSet[myMin][5], z1d10k[1:1,(i-1)*100+1:i*100])
        end
    end
    pyplot()
    for i in 1:10
        sampleTitle = @sprintf "Sample #%d" i
         pl = histogram( z1d10k[1,(i-1)*100+1:i*100], nbins = 30, normalize = true, color = :lightblue,label = sampleTitle)
         # for j in 1:10
         #     subSet = resultMatrix53[j,:]
         #     myMin = findmin(  [r[4]+(r[6]==:FORCED_STOP)*1e99 + (r[6]!=:FTOL_REACHED  && r[6]!=:XTOL_REACHED )*1e99 + dgPrune3(r,1,1e-100, 1e-2, true)*1e99 for r in subSet])
         #
         #     minParams =  subSet[myMin[2]][5]
         #     if i==j
         #         printGaussianParamsV3(minParams, 1, AngleAmplitude)
         #     end
         #     if myMin[1] < 1e99
         #         plotGaussesV3(minParams ,pl,-3:0.1:11, true, 1 + (i==j)*2)
         #     end
         #
         # end
         subSet = resultMatrix53[i,:]
         myMin = findmin(  [r[4]+(r[6]==:FORCED_STOP)*1e99 + (r[6]!=:FTOL_REACHED  && r[6]!=:XTOL_REACHED )*1e99 + dgPrune3(r,1,1e-100, 1e-2, true)*1e99 for r in subSet])
         plotGaussesV3(subSet[myMin[2]][5],pl,-3:0.1:11)

         Plots.display(pl)
         filename = @sprintf "histo on sample %d report-sigmacut-1e-10.eps" i
         savefig(pl, filename)
         wait_for_key(prompt="press key\n")
    end
    @save "resultMatrix-report-part2.jld2" crossValidationMatrixPrunedEntropy crossValidationMatrixPrunedQ

end

function brew(data::Array{Float64,2}, me::Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,1},Float64,Array{Float64,1},Symbol}, amplitudeCut = 1e-10, sigmaCut = 1e-40)
    localT = me
    nDim = size(data)[1]
    wastOfTimeCounter = 0
    forcedStopCounter = localT[6]==:FORCED_STOP
    while  forcedStopCounter<1 && wastOfTimeCounter<4 && dgPrune3(localT, nDim, amplitudeCut, sigmaCut, true)
        localTPert = dgPrune3(localT,nDim,amplitudeCut,sigmaCut,false)
        dgAParamSize = div(length(localTPert),2)
        #println(localTPert)
        localTPert[1:dgAParamSize] .+= 0.1*(rand(dgAParamSize))
        #println(localTPert)
        localT = QOptimizer(data, -1, [2000, 0, 10000], AngleAmplitude, localTPert , false)
        wastOfTimeCounter += 1
        forcedStopCounter += localT[6]==:FORCED_STOP || localT==:MAXEVAL_REACHED
    end

    return localT
end

makeReport()
