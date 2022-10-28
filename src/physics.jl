using LinearAlgebra

# SECTION: physical forces
F_mol(r, m1, m2, c1, c2; σ=1, ϵ=1) = (12*σ^12/norm(r)^13 - 6*σ^6/norm(r)^7) * (r/norm(r)) * 4ϵ
V_mol(r, m1, m2, c1, c2; σ=1, ϵ=1) = ((σ/norm(r))^12 - (σ/norm(r))^6) * 4ϵ

# rcap(r;c=10^-1) = norm(r) < c ? c/norm(r) * r : r
F_gra(r, m1, m2, c1, c2; G=1, rThresh=1) = norm(r) > rThresh ? -G*m1*m2/norm(r)^2 * (r/norm(r)) : [0, 0]
V_gra(r, m1, m2, c1, c2; G=1, rThresh=1) = norm(r) > rThresh ? -G*m1*m2/norm(r) : -G*m1*m2/norm(rThresh)

F_ele(r, m1, m2, c1, c2; Q=1, rThresh=1) = norm(r) > rThresh ? c1*c2 / norm(r)^2 * (r/norm(r)) : [0, 0]
V_ele(r, m1, m2, c1, c2; Q=1, rThresh=1) = norm(r) > rThresh ? c1*c2 / norm(r) : c1*c2 / norm(rThresh)

F_bil(r, m1, m2, c1, c2; σ=1, ϵ=1) = 48*ϵ*σ^12 / norm(r)^13 * (r/norm(r))
V_bil(r, m1, m2, c1, c2; σ=1, ϵ=1) = 4ϵ * (σ/norm(r))^12

function V_glb(r, m1, m2, c1, c2, s, d, A; rThresh=2)
    p = ( ((1-s)/abs(d))^(1/s) * s^((1-s)^-1) )^(A^-1)
    x = norm(r)
    if (x >= rThresh) || (d > 0)
        return s*abs(d)/d * (-abs(d)/(s-1))^(1-1/s) * (p*x)^-A - (p*x)^(-s*A)
    else
        return V_glb(rThresh, m1, m2, c1, c2, s, d, A; rThresh=rThresh)
    end
end

function F_glb(r, m1, m2, c1, c2, s, d, A, rThresh=2)
    p = ( ((1-s)/abs(d))^(1/s) * s^((1-s)^-1) )^(A^-1)
    x = norm(r)
    if (x >= rThresh) || (d > 0)
        α = s*abs(d)/d * (-abs(d)/(s-1))^(1-1/s)
        return (-p * A * α * (p*x)^-(A+1) + p*s*A*(p*x)^-(s*A+1)) * (r/x) * -1
    else
        # return V_glb(rThresh, m1, m2, c1, c2; s=s, d=d, A=A, rThresh=rThresh)
        return [0, 0]
    end
end

# bondVel(r, m) = 2*sqrt(1/m) * sqrt(-V_mol(r, 1, 1, 1, 1))

function getKE(v, m)
    return [m[i]*norm(v[i,:])^2 for i = 1:size(v)[1]] |> x->sum(x)/2
end

function getEnergy(x, v, m, c, pairEnergy, xB, yB)
    K = getKE(v, m)
    # print(K)
    V = 0
    for i = 1:size(x)[1]
        for j = i+1:size(x)[1]
            r = periodDistance(x[i,:], x[j,:], xB, yB)
            vv = pairEnergy(r, m[i], m[j], c[i], c[j])
            V += vv
        end
    end
    E = V + K
    return([E, V, K])
end

function getMomentum(v, m)
    return [sum(v[:,i] * m[i]) for i = 1:size(v)[2]]
end

function getTemperature(U, N)
    return U / (N - 0.5)
    #return getKE(v, m) / size(v)[1]
end

# produces vector from x2 to x1
function periodDistance(x1, x2, xB, yB)
    xAlt = (x2[1] < x1[1] ? -1 : 1) * (min(xB[2] - x1[1], x1[1] - xB[1]) + min(x2[1] - xB[1], xB[2] - x2[1]))
    yAlt = (x2[2] < x1[2] ? -1 : 1) * (min(yB[2] - x1[2], x1[2] - yB[1]) + min(x2[2] - yB[1], yB[2] - x2[2]))
    x = x1[1] - x2[1]
    y = x1[2] - x2[2]
    return [minf(abs, [x, xAlt]), minf(abs, [y, yAlt])]
end

function minf(f, l)
    l2 = f.(l)
    i = argmin(l2)
    return l[i]
end
