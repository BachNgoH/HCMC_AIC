import React, { useContext, useState } from "react";
import NextPageContext from "../store/nextpageCtx";
import classes from "./TextQueryForm.module.css";

function TextQueryForm({ setDataList }) {
    const [query, setQuery] = useState("");
    const [ocrQuery, setOCRQuery] = useState("");

    const nextpageCtx = useContext(NextPageContext);
    const [selectedImage, setSelectedImage] = useState();

    const fetch_image = async (url) => {
        const response = await fetch(
            `http://localhost:5000/${url}`
        );
        if (response.ok) {
            const data = await response.json();
            setDataList(data);
            nextpageCtx.setPage(1);
            nextpageCtx.setQuery(query);
            nextpageCtx.setPhotoId("");
        }
    };

    const submitHandler = async (e) => {
        e.preventDefault();
        await fetch_image(`api?q=${query}&page=${1}`);
    };

    const submitOCRHandler = async (e) => {
        e.preventDefault();
        console.log(ocrQuery);
        await fetch_image(`ocr?q=${ocrQuery}`)
    }

    const changeHandler = (event) => {
        setSelectedImage(event.target.files[0]);
    };

    const handleSubmission = (event) => {
        event.preventDefault();
        const formData = new FormData();

        formData.append("file", selectedImage);
        const fetch_image = async () => {
            const response = await fetch(`http://localhost:5000/photo`, {
                method: "POST",
                body: formData,
            });

            if (response.ok) {
                const data = await response.json();
                console.log(data);
                setDataList(data);
            }
        };
        fetch_image();
    };

    const handleKeyDown = (e) => {
        e.target.style.height = "inherit";
        e.target.style.height = `${e.target.scrollHeight}px`;
    };

    return (
        <>
            <form onSubmit={submitHandler} className={classes.form}>
                <label>Text Query</label>
                <textarea
                    className={classes.input}
                    onKeyDown={handleKeyDown}
                    placeholder="Type your query here ..."
                    onChange={(e) => setQuery(e.target.value)}
                />

                <button className={classes.scoreBtn}>Score</button>
            </form>
            <form className={classes.form}>
                <input
                    type="file"
                    name="file"
                    onChange={changeHandler}
                    accept="image/*"
                    className={classes.fileinput}
                />
                {selectedImage && <img src={URL.createObjectURL(selectedImage)}/>}
                <button className={classes.scoreBtn} onClick={handleSubmission}>
                    Score
                </button>
            </form>
            <form onSubmit={submitOCRHandler} className={classes.form}>
                <label>OCR Query</label>
                <textarea
                    className={classes.input}
                    onKeyDown={handleKeyDown}
                    placeholder="Type your query here ..."
                    onChange={(e) => setOCRQuery(e.target.value)}
                />

                <button className={classes.scoreBtn}>Score</button>
            </form>
        </>
    );
}

export default TextQueryForm;
