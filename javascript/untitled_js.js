function submit_imagegen() {
    var id = randomId();
    localSet("untitled_merger_task_id", id);

    requestProgress(id, gradioApp().getElementById('untitled_merger_gallery_container'), gradioApp().getElementById('untitled_merger_gallery'), function() {
        localRemove("untitled_merger_task_id");
    });

    var res = create_submit_args(arguments);

    res[0] = id;

    return res;
}