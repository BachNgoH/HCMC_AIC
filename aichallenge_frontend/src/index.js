import React from "react";
import ReactDOM from "react-dom/client";

import "./index.css";
import App from "./App";
import { SubmissionProvider } from "./components/store/submissionContext";
import { NextPageContextProvider } from "./components/store/nextpageCtx";

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
    <NextPageContextProvider>
        <SubmissionProvider>
            <App />
        </SubmissionProvider>
    </NextPageContextProvider>
);
