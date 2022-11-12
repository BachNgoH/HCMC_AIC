import React, { useContext, useEffect } from "react";
import Image from "./Image";
import { useState } from "react";
import classes from "./ImageList.module.css";
import NextPageContext from "../store/nextpageCtx";

function ImageList({
    dataList,
    sendKNNreq,
    setOpen,
    setOpenedVidID,
    setDataList,
}) {
    const [images, setImages] = useState([]);
    const nextpageCtx = useContext(NextPageContext);
    useEffect(() => {
        let counter = 0;

        const im = dataList["data"].map((item) => (
            <Image
                sendKNNreq={sendKNNreq}
                video_id={item.slice(0, 9)}
                frame_id={item.slice(10)}
                key={`${item}${++counter}`}
                setOpen={setOpen}
                setVidID={setOpenedVidID}
            />
        ));
        setImages(im);
    }, [dataList]);

    const onMoreImage = () => {
        const query = nextpageCtx.query;
        const page = nextpageCtx.page;
        const photoId = nextpageCtx.photoId

        const fetch_image = async () => {
            const response = await fetch(
                `http://localhost:5000/api?q=${query}&page=${page+1}&photoId=${photoId}`
            );
            if (response.ok) {
                const data = await response.json();
                if (data["data"] === "NO MORE DATA") {
                    alert("No more data");
                    return;
                }
                setDataList((prevData) => {
                    console.log({ data: [...prevData["data"], data["data"]] });
                    return { data: [...prevData["data"], ...data["data"]] };
                });
                nextpageCtx.setPage((page) => page + 1);
            }
        };
        fetch_image();
    };

    return (
        <div className={classes.container}>
            {images}
            {/* {(nextpageCtx.query !== "" || nextpageCtx.photoId !== "") && (
                <button className={classes.btn} onClick={onMoreImage}>
                    More Image
                </button>
            )} */}
        </div>
    );
}

export default ImageList;
