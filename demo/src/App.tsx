// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import { InferenceSession, Tensor } from "onnxruntime-web";
import React, { useContext, useEffect, useState } from "react";
import "./assets/scss/App.scss";
import { handleImageScale } from "./components/helpers/scaleHelper";
import { modelScaleProps } from "./components/helpers/Interfaces";
import { onnxMaskToImage } from "./components/helpers/maskUtils";
import { modelData } from "./components/helpers/onnxModelAPI";
import Stage from "./components/Stage";
import AppContext from "./components/hooks/createContext";
import npyjs from "npyjs";

// Define image, embedding and model paths
const IMAGE_PATH = "/assets/data/dogs.jpg";
const IMAGE_EMBEDDING = "/assets/data/dogs_embedding.npy";
const MODEL_DIR = "/model/sam_onnx_quantized_example.onnx";

const getErrorText = (error: unknown) => {
  if (error instanceof Error) return error.message;
  return String(error);
};

const App = () => {
  const {
    clicks: [clicks],
    image: [, setImage],
    maskImg: [, setMaskImg],
  } = useContext(AppContext)!;
  const [model, setModel] = useState<InferenceSession | null>(null); // ONNX model
  const [tensor, setTensor] = useState<Tensor | null>(null); // Image embedding tensor
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  // The ONNX model expects the input to be rescaled to 1024. 
  // The modelScale state variable keeps track of the scale values.
  const [modelScale, setModelScale] = useState<modelScaleProps | null>(null);

  // Initialize the ONNX model. load the image, and load the SAM
  // pre-computed image embedding
  useEffect(() => {
    let isMounted = true;

    const initApp = async () => {
      const failures: string[] = [];
      setIsLoading(true);
      setLoadError(null);

      const url = new URL(IMAGE_PATH, location.origin);
      const [loadedModel, loadedTensor] = await Promise.all([
        initModel().catch((error) => {
          failures.push(
            [
              `Unable to load ONNX model from ${MODEL_DIR}.`,
              "Place sam_onnx_quantized_example.onnx in demo/model/.",
              getErrorText(error),
            ].join("\n")
          );
          return null;
        }),
        loadNpyTensor(IMAGE_EMBEDDING, "float32").catch((error) => {
          failures.push(
            [
              `Unable to load embedding from ${IMAGE_EMBEDDING}.`,
              "Place dogs_embedding.npy in demo/src/assets/data/.",
              getErrorText(error),
            ].join("\n")
          );
          return null;
        }),
        loadImage(url).catch((error) => {
          failures.push(
            [`Unable to load image from ${IMAGE_PATH}.`, getErrorText(error)].join(
              "\n"
            )
          );
          return null;
        }),
      ]);

      if (!isMounted) return;

      setModel(loadedModel);
      setTensor(loadedTensor);
      setMaskImg(null);
      setLoadError(failures.length > 0 ? failures.join("\n\n") : null);
      setIsLoading(false);
    };

    initApp();

    return () => {
      isMounted = false;
    };
  }, []);

  const initModel = async () => {
    if (MODEL_DIR === undefined) return null;
    return InferenceSession.create(MODEL_DIR);
  };

  const loadImage = async (url: URL) =>
    new Promise<void>((resolve, reject) => {
      const img = new Image();
      img.src = url.href;
      img.onload = () => {
        const { height, width, samScale } = handleImageScale(img);
        setModelScale({
          height: height,  // original image height
          width: width,  // original image width
          samScale: samScale, // scaling factor for image which has been resized to longest side 1024
        });
        img.width = width; 
        img.height = height; 
        setImage(img);
        resolve();
      };
      img.onerror = () => reject(new Error(`Request failed for ${url.pathname}`));
    });

  // Decode a Numpy file into a tensor. 
  const loadNpyTensor = async (tensorFile: string, dType: "float32") => {
    const npLoader = new npyjs();
    const npArray = await npLoader.load(tensorFile);
    const tensor = new Tensor(dType, npArray.data, npArray.shape);
    return tensor;
  };

  // Run the ONNX model every time clicks has changed
  useEffect(() => {
    runONNX();
  }, [clicks]);

  const runONNX = async () => {
    try {
      if (
        model === null ||
        clicks === null ||
        tensor === null ||
        modelScale === null
      )
        return;
      else {
        // Preapre the model input in the correct format for SAM. 
        // The modelData function is from onnxModelAPI.tsx.
        const feeds = modelData({
          clicks,
          tensor,
          modelScale,
        });
        if (feeds === undefined) return;
        // Run the SAM ONNX model with the feeds returned from modelData()
        const results = await model.run(feeds);
        const output = results[model.outputNames[0]];
        // The predicted mask returned from the ONNX model is an array which is 
        // rendered as an HTML image using onnxMaskToImage() from maskUtils.tsx.
        setMaskImg(onnxMaskToImage(output.data, output.dims[2], output.dims[3]));
      }
    } catch (e) {
      console.log(e);
    }
  };

  return (
    <div className="relative h-screen w-screen bg-slate-950 text-slate-50">
      <Stage />
      {(isLoading || loadError) && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-950/88 px-6">
          <div className="max-w-2xl rounded-3xl border border-slate-800 bg-slate-900/95 p-8 shadow-2xl">
            <p className="text-xs font-semibold uppercase tracking-[0.35em] text-cyan-300">
              Segment Anything
            </p>
            <h1 className="mt-3 text-2xl font-semibold text-white">
              {loadError ? "Demo assets are missing" : "Loading demo assets"}
            </h1>
            <p className="mt-3 text-sm leading-6 text-slate-300">
              {loadError
                ? "The app is running, but it still needs the exported embedding and ONNX model before mask prediction can work."
                : "Preparing the image, embedding, and ONNX model for interactive segmentation."}
            </p>
            {loadError && (
              <pre className="mt-5 whitespace-pre-wrap rounded-2xl border border-amber-500/30 bg-amber-500/10 p-4 text-left text-sm leading-6 text-amber-100">
                {loadError}
              </pre>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
