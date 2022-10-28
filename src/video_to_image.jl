using VideoIO
using Random
using Images
using Printf
# crop using 40:760, 50:796

function extractImages(videoFilename, s, d, outputdir)
    video = VideoIO.load(videoFilename) |> collect
    start = length(video) / 2 |> floor |> Int
    for x = start:10:length(video)
        xCropped = video[x][40:760, 50:796]
        display(xCropped)
        println("Use this image? (y/N) ")
        choice = readline()
        if choice == "y"
            name = @sprintf("s_%.02f_d_%.02f.jpg", s, d)
            Images.save(joinpath(outputdir, name), xCropped)
            break
        end
    end
end

function processVideos(dir)
    outputdir = joinpath(dir, "images")
    mkdir(outputdir)
    r = r"glb_(?<s>[\d.]+)_d_(?<d>-?[\d.]+)\.mp4"
    files = []
    for f in readdir(dir)
        m = match(r, basename(f))
        if m != nothing
            push!(files, (joinpath(dir, f), parse(Float64, m["s"]), parse(Float64, m["d"])))
        end
    end
    display(files)
    for (i, file) in enumerate(files)
        @printf("On image %d/%d\n", i, length(files))
        extractImages(file..., outputdir)
    end
end
