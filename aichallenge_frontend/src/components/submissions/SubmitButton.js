import React, { useContext, useEffect, useState } from "react";
import CsvDownload from "react-csv-downloader";
import SubmissionContext from "../store/submissionContext";
import classes from "./SubmitButton.module.css";

const SubmitButton = () => {
    const submissionCtx = useContext(SubmissionContext);
    const [submissionCSV, setSubmissionCSV] = useState(null);
    const [reframedSubmission, setRefreamedSubmission] = useState(null)
    const [isLoading, setIsLoading] = useState(false);

    useEffect(() => {
        const frameIds = submissionCtx.columns["column-1"].frameIds;
        if (frameIds) {
            setSubmissionCSV(
                frameIds.map((frameId) => {
                    return {
                        video: frameId.slice(0, 9) + ".mp4",
                        frame: frameId.slice(10),
                    };
                }),
            );
        }
    }, [submissionCtx.columns["column-1"].frameIds]);
    const clearSubmissionsHandler = () => {
        submissionCtx.clearSubmissions();
    };

    const findMostSimilarSubmissionFrames = () => {
        const fetchKNN = async () => {

            setIsLoading(true)
            const response = await fetch(
                `http://localhost:5000/submission`,
                {
                    method: "POST",
                    "headers": {"Content-Type": "application/json"},

                    body: JSON.stringify({
                        "data": submissionCSV
                    })
                }
            );
            if (response.ok) {
                const data = await response.json()
                console.log(data["data"]);
                setRefreamedSubmission(data["data"]);
            }
            setIsLoading(false)
        };
        fetchKNN();
    }

    const columns = [
        {id: "video", displayName: "video"},
        {id: "frame", displayName: "frame"},

    ]

    return (
        submissionCSV && (
            <div className={classes.container}>
                <button
                    className={classes.clearBtn}
                    onClick={clearSubmissionsHandler}
                >
                    Clear
                </button>
                <button className={classes.addKNNBtn} onClick={findMostSimilarSubmissionFrames}>
                    {!isLoading ? "Add KNN" : "Loading ..."} 
                </button>
                <CsvDownload
                    filename="submission.csv"
                    separator=","
                    columns={columns}
                    datas={reframedSubmission}
                    noHeader={true}
                >
                    <button className={classes.submitBtn}>Export Submission</button>
                </CsvDownload>

            </div>
        )
    );
};

export default SubmitButton;
